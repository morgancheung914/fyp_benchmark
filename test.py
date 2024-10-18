import datasets

ds = datasets.load_from_disk('/research/d2/fyp24/hlcheung1/fyp/fyp_benchmark/responses/PubMedQA')
ds = ds.remove_columns(["context", "long_answer", "sys_content", "user_content"])
for i in range(5):
    print(ds[i])
    print("\n\n")