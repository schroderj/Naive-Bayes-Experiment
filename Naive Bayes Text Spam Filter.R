#this file uses naive bayes classification to create an SMS text message spam filter
#raw data is in 2 columns with no header.  1st column labels as spam or ham
#2nd column shows the text of the messages
#######################
#read in data
#-------------------
#read in the raw data
SMSSpamCollection <- read.delim("SMSSpamCollection.txt", header = FALSE)

#name the columns
colnames(SMSSpamCollection) <- c("Class", "Text")

#change the column containing text to character vectors
SMSSpamCollection$Text <- as.character(SMSSpamCollection$Text)
#--------------------

#clean data
#--------------------
#install tm package for text cleaning
library(tm)

#create the "corpus" object so text can be cleaned
sms_corpus <- VCorpus(VectorSource(SMSSpamCollection$Text))

#replace uppercase with lower case
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))

#remove all numbers from the text messages
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)

#remove stop words from the messages (as, and, or, you, me.... words that 
#don't tell us very much in the context of machine learning)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())

#remove punctuation
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)

#in order to "stem" words (turn "learning" into "learn") need snowballC package
library(SnowballC)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)

#remove the extra spaces produces by removing punctuation, etc.
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
#------------------------

#Data Preparation
#------------------------

#Split the data into words (tokenization)
#create a "Document Term Matrix" an object with rows as documents (messages)
#and columns as terms (words)
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)

#split into train and test data, 75% for training, 25% for testing
sms_dtm_train <- sms_dtm[1:2388, ]
sms_dtm_test <- sms_dtm[2389:3184, ]

#save 2 vectors with labels for the training and testing variables
sms_test_labels <- SMSSpamCollection[2389:3184, ]$Class
sms_train_labels <- SMSSpamCollection[1:2388, ]$Class

#install "wordcloud" so we can visualize data
library(wordcloud)

#create wordcloud for all of our data DATA VIZ
#wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

#create wordcloud for just spam and just ham DATA VIZ
spam <- subset(SMSSpamCollection, Class == "spam")
ham <- subset(SMSSpamCollection, Class == "ham")
#wordcloud(spam$Text, max.words = 40, scale = c(3, 0.5))
#wordcloud(ham$Text, max.words = 40, scale = c(3, 0.5))

#Find words that appear more than 5 times
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

#Find only words that occur at least 5 times... others are useless
sms_dtm_freq_train<- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

#convert counts of words into "yes" or "no" categorical variable
#Do this using apply() which applies to all 
#Use Source() function to access custom function
source("ConvertCounts.R")
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

#get the e1071 library containing the Naive Bayes algorithm
library(e1071)

#Train the Naive Bayes model
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

#make a prediction and store in variable
sms_test_pred <- predict(sms_classifier, sms_test)

#Use cross table to compare prediction to actual
library(gmodels)
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))

#make better prediction by setting laplace to equal 1 and then check performance
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicted', 'actual'))

