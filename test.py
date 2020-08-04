""" Shows how to use flask and matplotlib together.
Shows SVG, and png.
The SVG is easier to style with CSS, and hook JS events to in browser.
python3 -m venv venv
. ./venv/bin/activate
pip install flask matplotlib
python flask_matplotlib.py
"""
import io
import random
from flask import Flask, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG

from sklearn.decomposition import PCA #Grab PCA functions
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from gensim.models import Word2Vec
from wordcloud import WordCloud, STOPWORDS

from matplotlib.figure import Figure


app = Flask(__name__)

f2_india = open('india_wv', 'rb')      
word_vectors = pickle.load(f2_india) 

f2_india_score = open('india_score', 'rb')      
score = pickle.load(f2_india_score) 

def show_wordcloud(data, title = None):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))
    return wordcloud

def embedding(dish):
    r = np.zeros(400)
    for i,j in enumerate(dish):
        v = word_vectors.get_vector(j)
        r = r + v
    return r

def pca():
   
    f2_india_phrase_tokens = open('india_dishes', 'rb')      
    matching_dishes = pickle.load(f2_india_phrase_tokens)
    print(matching_dishes)
    pca_dishes = np.zeros((len(matching_dishes),400))
    print("HERE1")
    for i,j in enumerate(matching_dishes):
        pca_dishes[i] = embedding(j)
    print("HERE2")
    print(pca_dishes.shape)
    pca = PCA(n_components=2)
    result = pca.fit_transform(pca_dishes)
    return result, matching_dishes


@app.route("/")
def index():
    """ Returns html with the img tag for your plot.
    """
    num_x_points = int(request.args.get("num_x_points", 50))
    # in a real app you probably want to use a flask template.
    return f"""
    <h1>Flask and matplotlib</h1>
    <h2>Random data with num_x_points={num_x_points}</h2>
    <form method=get action="/">
      <input name="num_x_points" type=number value="{num_x_points}" />
      <input type=submit value="update graph">
    </form>
    <h3>Plot as a png</h3>
    <img src="/matplot-as-image-{num_x_points}.png"
         alt="random points as png"
         height="400"
    >
    <h3>Plot as a SVG</h3>
    <img src="/matplot-as-image-{num_x_points}.svg"
         alt="random points as svg"
         height="800"
    >
    """
    # from flask import render_template
    # return render_template("yourtemplate.html", num_x_points=num_x_points)


@app.route("/matplot-as-image-<int:num_x_points>.png")
def plot_png(num_x_points=50):
    """ renders the plot on the fly.
    """
    fig = Figure()
    fig.set_size_inches(12, 12)
    axis = fig.add_subplot(1, 1, 1)
    result, dishes = pca()
    axis.plot(result[:, 0], result[:, 1], 'o')
    for i, r in enumerate(result):
        axis.annotate(dishes[i][0]+' '+dishes[i][1],r , xytext = .95*r)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/matplot-as-image-<int:num_x_points>.svg")
def plot_svg(num_x_points=50):
    """ renders the plot on the fly.
    """
    fig = Figure()
    fig.set_size_inches(10, 10)
    axis = fig.add_subplot(1, 1, 1)
    result = show_wordcloud(score)
    print(type(result))
    axis.imshow(result)
    output = io.BytesIO()
    FigureCanvasSVG(fig).print_svg(output)
    return Response(output.getvalue(), mimetype="image/svg+xml")


if __name__ == "__main__":
    import webbrowser

    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)