# Goals for week 09

1. Practice working with regular expressions.
2. Practice working with stemming and lemmatization.
3. Work with `nltk`, `gensim` and `spaCy`.
4. Apply named-entity recognition.
5. Create and implement Naive Bayes classifier.

## Data Science

Learning how to model data effectively.

These tasks would require new packages. Please install them via the command `pip install -Ur requirements.txt`.

### Task 01

**Description:**

Take a look at the following string:

```python
test_string = "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"
```

Using regular expressions, output:

1. The sentences.
2. All capitalized words.
3. The string, split on spaces.
4. All numbers.

**Acceptance criteria:**

1. Regular expressions are used.

### Task 02

**Description:**

Here is the first scene of the [Monty Python's Holy Grail](https://en.wikipedia.org/wiki/Monty_Python_and_the_Holy_Grail):

```python
scene_one = "SCENE 1: [wind] [clop clop clop] \nKING ARTHUR: Whoa there!  [clop clop clop] \nSOLDIER #1: Halt!  Who goes there?\nARTHUR: It is I, Arthur, son of Uther Pendragon, from the castle of Camelot.  King of the Britons, defeator of the Saxons, sovereign of all England!\nSOLDIER #1: Pull the other one!\nARTHUR: I am, ...  and this is my trusty servant Patsy.  We have ridden the length and breadth of the land in search of knights who will join me in my court at Camelot.  I must speak with your lord and master.\nSOLDIER #1: What?  Ridden on a horse?\nARTHUR: Yes!\nSOLDIER #1: You're using coconuts!\nARTHUR: What?\nSOLDIER #1: You've got two empty halves of coconut and you're bangin' 'em together.\nARTHUR: So?  We have ridden since the snows of winter covered this land, through the kingdom of Mercea, through--\nSOLDIER #1: Where'd you get the coconuts?\nARTHUR: We found them.\nSOLDIER #1: Found them?  In Mercea?  The coconut's tropical!\nARTHUR: What do you mean?\nSOLDIER #1: Well, this is a temperate zone.\nARTHUR: The swallow may fly south with the sun or the house martin or the plover may seek warmer climes in winter, yet these are not strangers to our land?\nSOLDIER #1: Are you suggesting coconuts migrate?\nARTHUR: Not at all.  They could be carried.\nSOLDIER #1: What?  A swallow carrying a coconut?\nARTHUR: It could grip it by the husk!\nSOLDIER #1: It's not a question of where he grips it!  It's a simple question of weight ratios!  A five ounce bird could not carry a one pound coconut.\nARTHUR: Well, it doesn't matter.  Will you go and tell your master that Arthur from the Court of Camelot is here.\nSOLDIER #1: Listen.  In order to maintain air-speed velocity, a swallow needs to beat its wings forty-three times every second, right?\nARTHUR: Please!\nSOLDIER #1: Am I right?\nARTHUR: I'm not interested!\nSOLDIER #2: It could be carried by an African swallow!\nSOLDIER #1: Oh, yeah, an African swallow maybe, but not a European swallow.  That's my point.\nSOLDIER #2: Oh, yeah, I agree with that.\nARTHUR: Will you ask your master if he wants to join my court at Camelot?!\nSOLDIER #1: But then of course a-- African swallows are non-migratory.\nSOLDIER #2: Oh, yeah...\nSOLDIER #1: So they couldn't bring a coconut back anyway...  [clop clop clop] \nSOLDIER #2: Wait a minute!  Supposing two swallows carried it together?\nSOLDIER #1: No, they'd have to have it on a line.\nSOLDIER #2: Well, simple!  They'd just use a strand of creeper!\nSOLDIER #1: What, held under the dorsal guiding feathers?\nSOLDIER #2: Well, why not?\n"
```

Output:

- a tokenized version of the fourth sentence;
- the unique tokens in (or the `vocabulary` of) the entire scene;
- the start and end indices of the first occurrence of the word `coconuts` in the scene;
- all texts in square brackets in the first line;
- the script notation (e.g. `Character:`, `SOLDIER #1:`) in the fourth sentence.

**Acceptance criteria:**

1. Regular expressions are used.
2. A set is used to hold the vocabulary.

### Task 03

**Description:**

Twitter is a frequently used source for NLP text and tasks. In this exercise, you'll build a more complex tokenizer for tweets with hashtags and mentions. Here're example tweets that we'll work with:

```python
tweets = ['This is the best #nlp exercise ive found online! #python', '#NLP is super fun! <3 #learning', 'Thanks @datacamp :) #nlp #python']
```

Using the function `regexp_tokenize` and the class `TweetTokenizer` from the package `nltk`, output:

- All hashtags in first tweet.
- All mentions and hashtags in last tweet.
- All tokens.

**Acceptance criteria:**

1. The package `nltk` is used.
2. The module `re` is not used in the solutions.

### Task 04

**Description:**

Build a bag-of-words counter using a Wikipedia article that you can find in our `DATA` folder in the file `article.txt`. Before counting the tokens, lowercase each of them. Output the top 10 most common tokens in the bag-of-words.

Answer the following question in a comment: *By looking at the most common tokens, what topic does this article cover?*.

You'll quickly see that the most common tokens won't really help you answer this question. So, we'll have to perform some text preprocessing. Experiment with learned options until the answer becomes clearer.

Output the newly obtained top 10 most common tokens.

**Acceptance criteria:**

1. A comment is written with an answer to the question in the description.

### Task 05

**Description:**

Let's play with `gensim`. In our `DATA` folder you'll find `messy_articles.txt` - it contains a few articles from Wikipedia, which were preprocessed by lowercasing all words, tokenizing them, and removing stop words and punctuation. Load them into a list of lists and create a `gensim` dictionary and corpus.

Output:

1. The id of the word `computer`.
2. The first $10$ word ids in the fifth document with their frequency counts.
3. The $5$ most common words in the fifth document.
4. The top $5$ words across *all* documents alongside their counts.

Lastly, apply `gensim`'s class `TfidfModel` to the whole corpus and for the fifth document output:

- the first five term ids with their weights;
- the top five words.

**Acceptance criteria:**

1. All required outputs are present when the script is executed.

### Task 06

**Description:**

In the `article_uber.txt` file you'll find a sample news article. Output:

- the tokens in the last sentence alongside their part-of-speech tag;
- the first sentence as a chunked named-entity sequence. For this, use `nltk.ne_chunk_sents` and set `binary=True`. This would tag the named entities without specifying their exact types;
- the chunks that were tagged as named entities (i.e. have a label `NE`) in all sentences.

**Acceptance criteria:**

1. All required outputs are present when the script is executed.

### Task 07

**Description:**

Perform named-entity recognition on the `news_articles.txt` dataset (setting `binary` to `False`). Then, create a pie chart from the distribution of the various labels.

**Acceptance criteria:**

1. A pie chart is generated when the script is executed.

### Task 08

**Description:**

Let's now compare the differences between `nltk` and `spaCy`'s NER annotations. Load in the `article_uber.txt` data. Output all found entities and their labels.

Answer the following question in a comment: *Which are the extra categories that spacy uses compared to nltk in its named-entity recognition?*.

```text
A. GPE, PERSON, MONEY
B. ORGANIZATION, WORKOFART
C. NORP, CARDINAL, MONEY, WORKOFART, LANGUAGE, EVENT
D. EVENT_LOCATION, FIGURE
```

**Acceptance criteria:**

1. The required output is present when the script is executed.
2. The letter of the correct answer is written in a comment.

### Task 09

**Description:**

Let's build a text classifier that can predict whether a news article is fake or real. For this we'll use the file `fake_or_real_news.csv`.

Perform a data audit. Because the only features we have are text fields, the data audit would focus more on extracting useful statistics from them and creating plots that provide insight into the nature of the data, rather than the standard correlation measurements. Create statistics that are interesting to you and try to create useful plots that communicate insight to the client. Examples: `most common words in each class`, `class distribution`, `number of words per class` and similar that you deem would have added value.

Compare the performance between:

- at least two methods for preprocessing text.
- using a `CountVectorizer` and a `TfidfVectorizer`. For each perform hyperparameter tuning;
- keeping and removing stopwords;
- keeping and removing terms based on their document frequency using different thresholds;

For the best model, explore the words the classifier thinks are strong indicators for each class. Output the top $20$ words that have the highest empirical log probability for the classes. Also, output the usual classification metrics.

**Acceptance criteria:**

1. An Excel file showing the data audit is produced.
2. An Excel file showing the model report is produced.
3. The best words for fake news are shown for the best model.
4. The best words for real news are shown for the best model.
5. Hyperparameter tuning is performed.
6. Experiments with various text preprocessing methods are performed.

## Engineering

Building systems and implementing models.

### Task 01

**Description:**  

Add an implementation of a Multinomial Naive Bayes classifier in `ml_lib.naive_bayes`. The class should have:

- the following methods as a minimum: `fit`, `predict` and `score`;
- the following features as a minimum: `class_count_`, `class_log_prior_`, `feature_count_`, and `feature_log_prob_`.

The user should be able to set different values for the parameters `alpha` and `fit_prior` using a method `__init__`.

**Acceptance criteria:**

1. A class `MultinomialNB` is added to `ml_lib.naive_bayes`.

### Task 02

**Description:**

Use your model to recreate the best model in task 9.

**Acceptance criteria:**

1. It is shown that the implementation can be used to recreate the results obtained via `sklearn`'s `MultinomialNB`.
