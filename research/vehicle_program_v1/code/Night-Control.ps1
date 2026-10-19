param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [ValidateSet('Check','Start','Status','Stop','Resume')][string]$Action='Status',
    [switch]$Live
)
$ErrorActionPreference='Stop'
$nightPlan = Get-Content -LiteralPath $Plan -Raw -Encoding UTF8 | ConvertFrom-Json
$ops = Join-Path $PSScriptRoot 'night_ops.py'
$supervisor = Join-Path $PSScriptRoot 'night_supervisor.py'
if ($Action -in @('Check','Status','Stop')) {
    & $nightPlan.python -B $ops --plan $Plan --action $Action.ToLower()
    exit $LASTEXITCODE
}
if ($nightPlan.mode -eq 'api' -and -not $Live) { throw 'Live start requires -Live; preparation never starts model calls.' }
$preflight = if ($Action -eq 'Resume') { 'resume-check' } else { 'check' }
& $nightPlan.python -B $ops --plan $Plan --action $preflight
if ($LASTEXITCODE -ne 0) { throw 'Offline preflight failed.' }
$taskArgs = '"' + $supervisor + '" --plan "' + $Plan + '"'
if ($Action -eq 'Resume') { $taskArgs += ' --resume' }
$taskAction = New-ScheduledTaskAction -Execute $nightPlan.pythonw -Argument $taskArgs -WorkingDirectory $PSScriptRoot
$taskSettings = New-ScheduledTaskSettingsSet -Hidden -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 16) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$existing = Get-ScheduledTask -TaskName $nightPlan.task_name -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.Actions[0].Execute -ne $nightPlan.pythonw -or $existing.Actions[0].Arguments -notlike ('*"' + $Plan + '"*')) {
        throw 'Task name belongs to a different action.'
    }
    if ($existing.State -eq 'Running') { throw 'Night task is already running.' }
    Unregister-ScheduledTask -TaskName $nightPlan.task_name -Confirm:$false
}
# No time trigger: the human must explicitly request Start or Resume.
Register-ScheduledTask -TaskName $nightPlan.task_name -Action $taskAction -Settings $taskSettings -Principal $principal | Out-Null
Start-ScheduledTask -TaskName $nightPlan.task_name
Write-Output ('Started independent task: ' + $nightPlan.task_name)
