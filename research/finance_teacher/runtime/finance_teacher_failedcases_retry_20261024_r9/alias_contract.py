"""Exact R4 scientific alias policy, no new prompt change."""
from pathlib import Path
from campaign import load
original=load('R6_pinned_R4_alias',Path(__file__).resolve().parent.parent/'finance_teacher_program_coldstart_20261024_r4/alias_contract.py')
project_messages=original.project_messages
