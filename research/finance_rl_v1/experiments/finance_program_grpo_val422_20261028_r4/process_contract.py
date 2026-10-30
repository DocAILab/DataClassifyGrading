def is_training_writer(command,state,parent,evaluation):
    return parent.encode() in command and evaluation.encode() not in command and b'-c' not in command and state!='Z'
