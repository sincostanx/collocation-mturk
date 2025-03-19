from flask import Flask, render_template
import csv
import pandas as pd

app = Flask(__name__)

# src1 = [
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
# ],
# labels1 = ["xxx", "rock", "wall clock", "a traffic cone", "statue", "barbeque grill", "rock", "wall clock", "a traffic cone", "statue"]

# src2 = [
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/rice_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
#     "https://collocation2024.github.io/image-mturk/test/walkway_0001_Background.png",
# ],
# labels2 = ["aaa", "bbb", "wall clock", "a traffic cone", "statue", "barbeque grill", "rock", "wall clock", "a traffic cone", "statue"]

IMAGES_PER_HIT = 10
df = pd.read_csv("dataset750_candidate_reformat_github.csv")
images = df["img_path"].to_list()
print(images[0])
target_objects = df["target"].to_list()

@app.route('/')
def index():
    # data = load_data('data.csv')
    # print(images[:IMAGES_PER_HIT])
    data = {"src": images[:IMAGES_PER_HIT], "labels": target_objects[:IMAGES_PER_HIT]}
    return render_template('custom_mturk_ui.html', data=data, page_id=0)

@app.route('/page/<int:page_id>')
def page(page_id):
    if page_id >= len(images) // IMAGES_PER_HIT:
        return render_template('custom_mturk_ui_final.html')
    else:
        current_images = images[IMAGES_PER_HIT*(page_id) : IMAGES_PER_HIT*(page_id+1)]
        current_targets = target_objects[IMAGES_PER_HIT*(page_id) : IMAGES_PER_HIT*(page_id+1)]
        data = {"src": current_images, "labels": current_targets}
        return render_template('custom_mturk_ui.html', data=data, page_id=page_id)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=4444)