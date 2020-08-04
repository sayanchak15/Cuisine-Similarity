# Cuisine-Similarity
Find cuisine similarities from user review data


This application is an effort of extracting dish similarities from review text. For a person who is new to a particular cuisine – now can extract information about cuisines and which dishes would be favorable to him/her. For example – if a vegetarian person wants to know veggie options – that’s possible now with this application. All he/she has to do is select veggie food known to him. This application will find similar dishes for him.
Below are the user actions -
In this Application user will analyze specific cuisine by their Yelp review text. Here are the user steps -
1. User will select a cuisine
     2. Application will scan all Yelp reviews and extract quality phrases with AutoPhrase mining technique
3. User will select a dish from quality phrases
 
 4. Application will display a map with related dishes.
a. The first graph will cluster similar food in vector space. Unrelated foods
should be sparsely placed.
b. 2nd graph would show popularity measures with similar food. Bigger fonts
are more popular,
 interface design:
 • • • •
working URL:
The application was built in Flask micro framework.
Server hosted in AWS EC2 in Ubuntu with apache2.
Client side was built in simple HTML and CSS.
Client call to server side was with Flask APIs. The APIs used here are for -
o Extract Reviews
o Extract Quality phrases with novel Autophrase
o Use word embedding techniques (word2vec) to find similar dishes that
user selects. Plot the dishes in reduced 2 dimensional vector space using
PCA
o Using similarity scores and autophrase ranking – find popular dishes that
closely resembles. Plot wordcloud on them.
http://sayanchak.com
   
Usefulness of the application
For a person who is new to a particular cuisine – now can extract information about cuisines and which dishes would be favorable to him/her. For example – if a vegetarian person wants to know veggie options – that’s possible now with this application. All he/she has to do is select veggie food known to him. This application will find similar dishes for him.
Novelty of the application
Novel functions this application may address –
• Given a known dish, this application can find similar dishes which may be unknown to users.
• Among the group of dishes this application can find the popularity.
