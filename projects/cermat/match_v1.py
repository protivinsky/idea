import json
import datetime as dt
from admissions import AdmissionData, DeferredAcceptance
import pandas as pd
from projects.cermat.lib import StepLogger, data_root

json_path = data_root() / "input" / "export-pro-vysledky-rozrazeni-test.json"
raw_data = json.load(open(json_path, "r"))
df = pd.DataFrame(raw_data)

num_students_full = df["rcu"].drop_duplicates().shape[0]
num_students_full  # 157_019

df.shape
df.head()
df.dtypes
df.iloc[0]

# NOTE:  ===  DATA  ===
# loading is easy, the columns are likely:
# - rcu = ID uchazece
# - sofId = ID oboru
# - priorita
# - poradi
# - kapacita

# NOTE:  ===  CHECK THE DATA  ===

# kazdy obor ma unikatni kapacitu
assert df["sofId"].drop_duplicates().shape[0] == df[["sofId", "kapacita"]].drop_duplicates().shape[0]
# kazdy radek je prihlaska - neshoduje se id zaka a oboru ani s prioritou
assert df.shape[0] == df[["rcu", "sofId"]].drop_duplicates().shape[0]
assert df.shape[0] == df[["rcu", "priorita"]].drop_duplicates().shape[0]
# zaci na stejnem oboru nemaji stejne poradi
assert df.shape[0] == df[["sofId", "poradi"]].drop_duplicates().shape[0]
# nejsou duplicitni prihlasky (zak, obor)
assert df.shape[0] == df[["rcu", "sofId"]].drop_duplicates().shape[0]

df.shape[0]
df[["sofId", "poradi"]].drop_duplicates().shape[0]

foo = df[["sofId", "poradi"]].value_counts().reset_index()
foo["count"].value_counts()

assert foo.iloc[0]["count"] == 32
weird_sofId = foo.iloc[0]["sofId"]
weird_sofId

df[df["sofId"] == weird_sofId]

# FIXME:
# dae2eea1-caf8-44e0-901b-05dc4b2c42ec
# REDIZO: 600010490
# Střední průmyslová škola technická, Jablonec nad Nisou, Belgická 4852, příspěvková organizace
# --
# Uvadi poradi pro prvnich 50, nasledne davaji poradi 1_000_000.
# Mezi prvnimi 50 je 16 priorit 1; kapacitu maji 24. Dost pravdepodobne se zaplni.
# Predpokladejme, ze 1_000_000 znamena automaticke odmitnuti -> tyto prihlasky tedy mohou byt vyrazeny.
# Nasledne bych ale mel overit, ze zde byla zaplnena kapacita.
# Pokud nikoli, pak je toto problem a potrebuji to plne overit.

df["poradi"].value_counts().reset_index().sort_values("poradi")
# looks ok otherwise

df.shape # (419242, 12)
df = df[df["poradi"] != 1_000_000].copy()
df.shape # (419210, 12)

# NOTE:  ===  EVALUATION  ===
# Ok, with the exception of the weird SPŠT above, all is fine.
# Create the admission data and evaluate. Then do some basic validation over the result.

apps = df.sort_values(by="priorita").groupby("rcu")["sofId"].apply(list).to_dict()
exams = df.sort_values(by="poradi").groupby("sofId")["rcu"].apply(list).to_dict()
seats = df[["sofId", "kapacita"]].drop_duplicates().set_index("sofId")["kapacita"].to_dict()

# validate apps and exams are correct
# APPS
df["tmp"] = df.apply(lambda row: (row["sofId"], row["priorita"]), axis=1)
foo = df.sort_values(by="priorita").groupby("rcu")["tmp"].apply(list).to_dict()
for k, v in foo.items():
    i = v[0][1]
    for _, j in v[1:]:
        assert i < j
        i = j
    assert apps[k] == [x[0] for x in v]

# EXAMS
df["tmp"] = df.apply(lambda row: (row["rcu"], row["poradi"]), axis=1)
foo = df.sort_values(by="poradi").groupby("sofId")["tmp"].apply(list).to_dict()
for k, v in foo.items():
    i = v[0][1]
    for _, j in v[1:]:
        assert i < j
        i = j
    assert exams[k] == [x[0] for x in v]

df = df.drop(columns=["tmp"])

input_data = AdmissionData(applications=apps, exams=exams, seats=seats)
mech = DeferredAcceptance(data=input_data, logger=StepLogger())

now = dt.datetime.now()
res = mech.evaluate()
print(f"elapsed time: {dt.datetime.now() - now}")
# 6s

# Do I need to worry about SPST Jablonec? --> yes... they accepted only 18.
len(res.accepted[weird_sofId])

num_students = df["rcu"].drop_duplicates().shape[0]
num_students  # 157_019

len(res.rejected)
# 27_027 rejected

# FIXME:
# SPŠT Jablonec nad Nisou
# -> include even that 1_000_000 poradi, in order of scores

raw_data = json.load(open(json_path, "r"))
df = pd.DataFrame(raw_data)

apps = df.sort_values(by="priorita").groupby("rcu")["sofId"].apply(list).to_dict()
exams = df.sort_values(by="poradi").groupby("sofId")["rcu"].apply(list).to_dict()
seats = df[["sofId", "kapacita"]].drop_duplicates().set_index("sofId")["kapacita"].to_dict()

spst_id = "dae2eea1-caf8-44e0-901b-05dc4b2c42ec"
sim_prg.set_index("id_oboru").loc[spst_id]  # obor_kod = 26-41-L/01


spst = df[df["sofId"] == spst_id].copy()
spst_a = spst[spst["poradi"] != 1_000_000].sort_values("poradi").copy()
spst_b = spst[spst["poradi"] == 1_000_000].sort_values("body", ascending=False).copy()
spst_a
spst_b
spst_exam = spst_a["rcu"].tolist() + spst_b["rcu"].tolist()

exams[spst_id] = spst_exam

input_data = AdmissionData(applications=apps, exams=exams, seats=seats)
mech = DeferredAcceptance(data=input_data, logger=StepLogger())

now = dt.datetime.now()
res = mech.evaluate()
print(f"elapsed time: {dt.datetime.now() - now}")
# 6s

len(res.accepted[weird_sofId])

num_students = df["rcu"].drop_duplicates().shape[0]
num_students  # 157_019

len(res.rejected)
# 27_025 rejected



foo[k]
k = list(foo.keys())[0]
k
foo[k]


foo = df.set_index("
for k, v in apps.items():
    foo = 
    print(k)

apps


# NOTE:  ===  MESS  ===

df["rcu"].drop_duplicates().shape

sim = pd.read_csv(data_root() / "sim" / "df.csv")
sim_list_cols = ["izo", "obor_kod", "kapacita", "id_oboru"]
for c in sim_list_cols:
    sim[c] = sim[c].apply(eval)

sim_obory = sim["id_oboru"].explode().drop_duplicates()
sim_obory

sim_prg = pd.read_csv(data_root() / "sim" / "prg.csv")
sim_prg

df["sofId"].drop_duplicates().shape
df[["sofId", "kapacita"]].drop_duplicates().shape

sim.shape
sim.iloc[0]
sim["id_oboru"].iloc[0]


len(res)
res[:10]
