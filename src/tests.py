from functions import *
id = 7112
answer = get_data(id, "a")[0]
print(get_data(id, "q")[0])
print(
    "%"+str(jaccard_similarity(
        preprocess_text("İstanbul Boğazı köprü inşası 1931 yılında başlatıldı"),
        preprocess_text(answer),
    )) 
)