# TODO refactor it later
import os
import pandas as pd
import sys

def individual_file_to_commandline(ind_file):

    ind = []
    with open(ind_file, "r") as fp :
        line = fp.readlines()[0]
        tokens = line.split(",")
        ind.append(float(tokens[3][2:])) # store first token as float, ignoring first character "["
        for t in range(4, 4+20) :
            ind.append(float(tokens[t]))

    #if ind is Individual get only the num_weights! else check that ind is a list of num_weights
    param = ""
    for e in ind:
        param = param+str(e)+"#"
    param = param[:-1]
    return param

# as there are some issues with the SabberStone C# projects
sabberstone_base_dir = os.path.abspath("../../SabberStone/core-extensions/SabberStoneCoreAi")
os.chdir(sabberstone_base_dir)

DECKS = ["RenoKazakusMage", "MidrangeJadeShaman" , "AggroPirateWarrior"]
HERO_BY_DECK = {"RenoKazakusMage":"MAGE", "MidrangeJadeShaman":"SHAMAN" , "AggroPirateWarrior":"WARRIOR"}

base_cmd_line = "dotnet run --project SabberStoneCoreAi.csproj"

# read individual and convert it to string
INDIVIDUAL_BY_DECK = dict()
INDIVIDUAL_BY_DECK["RenoKazakusMage"] = "results/inspyred-individuals-file-07092018-153823-ONLY-MAGE.csv"
INDIVIDUAL_BY_DECK["MidrangeJadeShaman"] = "results/inspyred-individuals-file-07092018-153850-ONLY-SHAMAN.csv"
INDIVIDUAL_BY_DECK["AggroPirateWarrior"] = "results/inspyred-individuals-file-07092018-153819-ONLY-WARRIOR.csv"

d1 = "AggroPirateWarrior"
d2 = "MidrangeJadeShaman"
number_of_games = 20

# rest of the command line
cmd_line = base_cmd_line + " " + d1 + " " + HERO_BY_DECK[d1] + " " + individual_file_to_commandline(INDIVIDUAL_BY_DECK[d1]) + " "
cmd_line += d2 + " " + HERO_BY_DECK[d2] + " " + individual_file_to_commandline(INDIVIDUAL_BY_DECK[d2]) + " "
cmd_line += str(number_of_games) + " > ../../../alphazero/examples/HS/cmd_out.txt"

print(cmd_line)
os.system(cmd_line)
