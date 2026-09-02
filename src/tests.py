from functions import *

answer = get_data(id, "a")[0]

print(
    jaccard_similarity(
        preprocess_text("1931 yılında başlatıldı"),
        preprocess_text(answer),
    )
)