import re
import json
import datetime as dt
import pandas as pd
from admissions import AdmissionData, DeferredAcceptance
from projects.cermat.lib import StepLogger, data_root


file_path = data_root() / "audit-log-prod.txt"


def int_from_line(line: str) -> int:
    return int(re.search(r"\d+", line).group())


def parse_student_data(student_list):
    students = []
    for student in student_list.split(','):
        student_id, details = student.strip().split('[')
        rank, order = details[:-1].split('|')
        students.append({
            'student_id': int(student_id),
            'rank': "X" if rank == "X" else int(rank),
            'order': int(order)
        })
    return students


seats = {}
exams = {}
apps = {}
last_prio = {}
rejected = set()
accepted = {}
num_rejected = 0

current_step = 0
school_id = None
with open(file_path, "r") as file:
    for i, line in enumerate(file):
        line = line.strip()
        if not line:
            continue

        elif line == "===  SUMAR ===":
            break

        elif line.startswith("===  CYKLUS"):
            step_number = int_from_line(line)
            assert step_number == current_step + 1
            current_step = step_number

        elif line.startswith("Obor:"):
            school_id = line.split(';')[0].split(':')[1].strip()
            if school_id not in exams:
                exams[school_id] = set()
            if school_id not in accepted:
                accepted[school_id] = set()

        elif line.startswith("Zaplneni/Kapacita"):
            num_seats = int(line.split('/')[2])
            if school_id in seats:
                assert seats[school_id] == num_seats, f"L{i} = {line}"
            else:
                seats[school_id] = num_seats

        elif line.startswith("Pridani"):
            student_data = line.split(":")[1].strip()
            students = parse_student_data(student_data)
            for s in students:
                exams[school_id].add((s["student_id"], s["order"]))
                if s["student_id"] not in apps:
                    apps[s["student_id"]] = set()
                apps[s["student_id"]].add((school_id, s["rank"]))
                last_prio[s["student_id"]] = s["rank"]
                accepted[school_id].add(s["student_id"])

        elif line.startswith("Nevesli se, posouva se priorita"):
            student_data = line.split(":")[1].strip()
            students = parse_student_data(student_data)
            for s in students:
                exams[school_id].add((s["student_id"], s["order"]))
                if s["student_id"] not in apps:
                    apps[s["student_id"]] = set()
                apps[s["student_id"]].add((school_id, s["rank"] - 1))
                last_prio[s["student_id"]] = s["rank"] - 1

        elif line.startswith("Definitivne neprijati"):
            num_rejected += int_from_line(line.split(":")[0].strip())
            student_data = line.split(":")[1].strip()
            students = parse_student_data(student_data)
            for s in students:
                exams[school_id].add((s["student_id"], s["order"]))
                if s["student_id"] not in apps:
                    apps[s["student_id"]] = set()
                assert s["rank"] == "X"
                guess_rank = 1 if s["student_id"] not in last_prio else last_prio[s["student_id"]] + 1
                apps[s["student_id"]].add((school_id, guess_rank))
                rejected.add(s["student_id"])

        elif line.startswith("Odstraneni"):
            student_data = line.split(":")[1].strip()
            students = parse_student_data(student_data)
            for s in students:
                exams[school_id].add((s["student_id"], s["order"]))
                if s["student_id"] not in apps:
                    apps[s["student_id"]] = set()
                assert s["rank"] == last_prio[s["student_id"]]
                assert (school_id, s["rank"]) in apps[s["student_id"]]
                accepted[school_id].remove(s["student_id"])

        else:
            raise ValueError(f"Unrecognized line: {line}")

num_rejected
len(rejected)  # 23 948
len(apps)      # 152 089
len(exams)     # 6 048
len(seats)     # 6 048

sum_acc = 128141
sum_acc_diff = sum_acc - sum([len(v) for v in accepted.values()])
sum_acc_diff  # 0
sum_total = 157004
sum_total_diff = sum_total - len(apps)
sum_total_diff  # 4 915
sum_rej = 28863
sum_rej_diff = sum_rej - len(rejected)
sum_rej_diff  # 4 915

# NOTE: Who are the missing students here?
all_students = set(apps.keys())
len(all_students)  # 152_089
all_sofid = set(exams.keys())
len(all_sofid)  # 6_048

# load the previous json data
json_path = data_root() / "input" / "export-pro-vysledky-rozrazeni-test.json"
raw_data = json.load(open(json_path, "r"))
df = pd.DataFrame(raw_data)

df.iloc[0]
df["in_log"] = df["rcu"].isin(all_students)
df["in_log"].value_counts()
df["sof_in_log"] = df["sofId"].isin(all_sofid)
df["sof_in_log"].value_counts()

miss = df[~df["in_log"]].copy()
miss_sofid = df[~df["sof_in_log"]].copy()
len(miss["rcu"].drop_duplicates())
miss["priorita"].value_counts()
miss["poradi"].value_counts()
miss
miss["redizo"].value_counts()

miss["sofId"].drop_duplicates().shape
miss_sofid["sofId"].drop_duplicates().shape
miss_sofid

miss["redizo"].value_counts()
all_redizo = set(miss["redizo"])

redizo_spsejecna = 600004783
redizo_spsejecna in all_redizo
redizo_sssvt = 600006239
redizo_sssvt in all_redizo
redizo_sokolska = 600013936
redizo_sokolska in all_redizo
redizo_krizik = 600020151
redizo_krizik in all_redizo
redizo_chem = 600004678
redizo_chem in all_redizo


foo = list(apps.values())[10079]

apps2 = {}
exams2 = {}

for k, v in apps.items():
    apps2[k] = [ii for ii, _ in sorted(v, key = lambda x: x[1])]

for k, v in exams.items():
    exams2[k] = [ii for ii, _ in sorted(v, key = lambda x: x[1])]

input_data = AdmissionData(applications=apps2, exams=exams2, seats=seats)
mech = DeferredAcceptance(data=input_data, logger=StepLogger())
res = mech.evaluate()

len(res.rejected)
rejected == res.rejected
accepted == res.accepted


[x for x, i in sorted(foo, key = lambda x: x[1])]
foo


450491 in rejected

sum([len(v) for k, v in accepted.items()]) + len(rejected)

foo = []
for sch in seats3:
    if sch not in seats:
        foo.append(sch)

len(seats3) - len(seats)
len(foo)



line

seats

parsed_data = parse_file(file_path)

foo = {1, 2, 3}
foo.add(4)
foo
foo = {}
type(foo)
