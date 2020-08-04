from flask import Flask, render_template, request, session, make_response, Response
import itertools
from matplotlib.figure import Figure
from sklearn.decomposition import PCA #Grab PCA functions
import numpy as np
import pickle
from gensim.models import Word2Vec
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from wordcloud import WordCloud, STOPWORDS
import random

app = Flask(__name__)

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

def embedding(dish, word_vectors):
    r = np.zeros(400)
    for i,j in enumerate(dish):
        v = word_vectors.get_vector(j)
        r = r + v
    return r

def pca(matching_dishes, word_vectors):
    print(matching_dishes)
    pca_dishes = np.zeros((len(matching_dishes),400))
    print("HERE1")
    for i,j in enumerate(matching_dishes):
        pca_dishes[i] = embedding(j, word_vectors)
    print("HERE2")
    print(pca_dishes.shape)
    pca = PCA(n_components=2)
    result = pca.fit_transform(pca_dishes)
    return result, matching_dishes

def list_of_words(input_choice):
    result = input_choice.split()
    return result
   
def cloud_phrases(item):
    tmp = '_'
    for i in item:
        tmp = tmp+'_'+i
    return tmp

def get_phrases(text, thresold):
    best_matches_phrases = []
    best_matches_tokens = []
    for t in text:
        tmp = ''
        tmp1 = []
        if float(t[0]) >= thresold:
            
            for i in range(1,len(t)):
                tmp1.append(t[i])
                if tmp == '':
                    tmp = t[i]
                else: tmp = tmp + ' '+t[i]
            best_matches_phrases.append(tmp)
            best_matches_tokens.append(tmp1)
    return best_matches_phrases, best_matches_tokens


@app.route("/")
def hello():
   return render_template("index.html")



@app.route('/reviews', methods=['POST'])
def handle_data():
   cuisine = request.form['cuisine']
   # your code
   text = []
   result = []
   print('reviews/'+cuisine+'.txt')
   with open('reviews/'+cuisine+'.txt', 'r') as f:
      for line in f.readlines():
         text.append(line)
#  result = [text[0], text[1], text[2], text[3], text[4]]
   for i in random.sample(range(0, 300), 10):
      if text[i]=='\n':
        continue
      else: 
        result.append(text[i])
   ptext = []
   with open('autophrase/'+cuisine+'_phrases.txt', 'r') as f:
        for line in f.readlines():
            ptext.append(line.split())
   best_matches_phrases, top_phrases_tokens = get_phrases(ptext, .95)
   pd_top_phrase_tokens = open('top_phrase_tokens', 'wb')
   pickle.dump(top_phrases_tokens, pd_top_phrase_tokens)
   pd_top_phrase_tokens.close()
   resp = make_response(render_template("reviews.html",cuisine = cuisine, result = result
                        , phrases = best_matches_phrases, total_phrase = len(best_matches_phrases)))
   resp.set_cookie('cuisine', cuisine)
   return resp
   

@app.route('/cuisinemap', methods=['POST'])
def cuisinemap():
   cuisine = request.cookies.get('cuisine')
   pl_wv = open(cuisine+'_wv', 'rb')      
   word_vectors = pickle.load(pl_wv) 
   input_phrase = request.form['phrase']
   pl_top_phrase_tokens = open('top_phrase_tokens', 'rb')   
   matching_dishes = []
   score ={}   
   count = 0
   top_phrases_tokens = pickle.load(pl_top_phrase_tokens) 
   for item in top_phrases_tokens:
      try:
         s_sim = word_vectors.n_similarity(list_of_words(input_phrase), item)
      except KeyError:
        continue
      except ZeroDivisionError:
        continue
      if s_sim > .5 :
   #     or s2> thr or s3> thr:
         matching_dishes.append(item)
         score[cloud_phrases(item)]=s_sim*10
         count += 1
   result, dishes = pca(matching_dishes, word_vectors)
   f1_result= open('result', 'wb')
   pickle.dump(result, f1_result)
   f1_result.close()
   f1_dishes= open('dishes', 'wb')
   pickle.dump(dishes, f1_dishes)
   f1_dishes.close()
   f1_score= open('score', 'wb')
   pickle.dump(score, f1_score)
   f1_score.close()
   return render_template("cuisinemap.html", result = result, dishes = dishes)



@app.route("/pca-image.png")
def plot_pca_png():
    """ renders the plot on the fly.
    """
   #  result = np.fromstring(result, dtype=np.float, sep='] [')
    f2_result = open('result', 'rb')      
    result = pickle.load(f2_result)
    f2_dishes = open('dishes', 'rb')      
    dishes = pickle.load(f2_dishes)
    print("Inside PLOT", result)
    fig = Figure()
    fig.set_size_inches(12, 12)
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(result[:, 0], result[:, 1], 'o')
    for i, r in enumerate(result):
        axis.annotate(cloud_phrases(dishes[i]),r , xytext = .95*r)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")

@app.route("/wc-image.png")
def plot_wc_png():
   f2_score = open('score', 'rb')      
   score = pickle.load(f2_score)
   print('score', score)
   fig = Figure()
   fig.set_size_inches(20, 20)
   axis = fig.add_subplot(1, 1, 1)
   result = show_wordcloud(score)
   print(type(result))
   axis.imshow(result)
   output = io.BytesIO()
   FigureCanvasSVG(fig).print_svg(output)
   return Response(output.getvalue(), mimetype="image/svg+xml")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=80)