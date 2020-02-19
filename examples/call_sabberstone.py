# TODO refactor it later

sabberstone_base_dir = "../../SabberStone"
DECKS = ["RenoKazakusMage", "MidrangeJadeShaman" , "AggroPirateWarrior"]
HERO_BY_DECK = {"RenoKazakusMage":"MAGE", "MidrangeJadeShaman":"SHAMAN" , "AggroPirateWarrior":"WARRIOR"}

base_cmd_line = "dotnet run --project " + sabberstone_base_dir + "/core-extensions/SabberStoneCoreAi/SabberStoneCoreAi.csproj"

# read individual and convert it to string
INDIVIDUAL_BY_DECK = dict()
INDIVIDUAL_BY_DECK["RenoKazakusMage"] = "/core-extensions/SabberStoneCoreAi/results/inspyred-individuals-file-07092018-153823-ONLY-MAGE.csv"
INDIVIDUAL_BY_DECK["MidrangeJadeShaman"] = "/core-extensions/SabberStoneCoreAi/results/inspyred-individuals-file-07092018-153850-ONLY-SHAMAN.csv"
INDIVIDUAL_BY_DECK["AggroPirateWarrior"] = "/core-extensions/SabberStoneCoreAi/results/inspyred-individuals-file-07092018-153819-ONLY-WARRIOR.csv"

d1 = "AggroPirateWarrior"
d2 = "MidrangeJadeShaman"

# rest of the command line
cmd_line = base_cmd_line + " " + d1 + " " + HERO_BY_DECK[d1] + " "
cmd_line += d2 + " " + HERO_BY_DECK[d2] + " "

print(cmd_line)
