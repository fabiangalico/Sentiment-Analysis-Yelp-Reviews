###################################################################################################
### Sentiment Analysis: YELP Reviews (Group 8) ###################################################
### By: Fabian Galico, Ajay Parihar, Patrick Dundon ##############################################
### Our focus: Evaluating sentiment of Yelp reviews pertaining to restaurants in Las Vegas, NV ###
#################################################################################################

#######################################################
########## 1. READ DATA & CREATE BASETABLE ############
#######################################################

#import libraries
install.packages('tidytext')
install.packages('ggraph')
install.packages('radarchart')
install.packages('tm')
library(data.table)
library(dplyr)
library(tidytext)
library(tidyr)
library(igraph)
library(grid)
library(ggraph)
library(radarchart)
if (!require("wordcloud")) {
  install.packages("wordcloud",repos="https://cran.rstudio.com/",
                   quiet=TRUE)
  require("wordcloud")
}
install.packages("tm")
library("tm")
library("SnowballC")
library("caTools")
library("rpart")
install.packages("rpart.plot")
library("rpart.plot")
if (!require("ROCR")) install.packages("ROCR", quiet=TRUE) ; require("ROCR")
library(ROCR)
if (!require("randomForest")) install.packages("randomForest", quiet=TRUE) ; require("randomForest")
library(randomForest)
if (!require("irlba")) install.packages("irlba", quiet=TRUE) ; require("irlba")
install.packages("sentimentr")
library(sentimentr)
install.packages("ggplot2")
library(ggplot2)
require(plyr)
library(RWeka)
Sys.setenv(JAVA_HOME="C:/Program Files/Java/jdk-11.0.2/")
install.packages("rJava")
library(rJava)
install.packages("pROC")
library(pROC)


#read files containing Yelp businesses and Yelp reviews
businesses <- fread("yelp_business.csv", nrows = 100000)
reviews <- fread("yelp_review.csv", nrows = 1000000)

#merge businesses and reviews tables and filter for 'Las Vegas' and 'Restaurant'
vegasrest <- reviews %>% 
  left_join(businesses, by = "business_id") %>% 
  filter(city == 'Las Vegas') %>%
  filter(categories %like% 'Restaurant') %>%
  filter(useful > 3) # Only reviews that at least 3 people found useful are taken

# All text is converted to lower case
vegasrest$text <- tolower(vegasrest$text)

#save the dataframe
save(vegasrest,file="vegasrest.Rda")


################################
##### 2. DATA EXPLORATION #####
##############################

#load the dataframe
load("vegasrest.Rda")

#wordcloud for words in reviews
splitwords <- unlist(strsplit(vegasrest$text, " "))

wordcloud(splitwords, 
          min.freq = 300,
          random.order=FALSE, 
          rot.per=0.35,
          colors=brewer.pal( 8,"Dark2"))

#wordcloud for categories
categories <- unlist(strsplit(vegasrest$categories, ";"))

remove_categories <- c("Restaurants", "Food", "Services")
clean_categories <- removeWords(categories, remove_categories)

wordcloud(clean_categories, 
          min.freq = 300,
          random.order=FALSE, 
          rot.per=0.35,
          colors=brewer.pal( 8,"Dark2"))

#############################################################
######### 3. SENTIMENT ANALYSIS - FIRST INSIGHTS ############
#############################################################

# Stop words and repeating words like Las vegas, http, www.yelp.com are removed
vegasrest$text <- removeWords(vegasrest$text ,c(stopwords("en"), "vegas","http","www.yelp.com" ))
vegasrest$text <- removeNumbers(vegasrest$text)

#######
####### CREATE BIGRAMS TO MAP RELATIONSHIPS BETWEEN WORDS ########
#################################################################

# Bigrams are created with words in review text
bigrams <- vegasrest %>% 
  unnest_tokens(bigram, text, token = "ngrams", n = 2)

# List of words considered significant
analysis_word <- c("food", "place", "service")

# Creates data for network analysis graph
bg_grapgh <- bigrams %>%
  # Words from bigram are seperated
  separate(bigram, c("word1", "word2"), sep = " ")  %>% 
  # Count for each combination of words are calculated
  group_by(word1, word2) %>% 
  summarise( n = n()) %>%
  # Only the combination with significant words in the beginning and min freq of 30 are taken
  filter( word1 %in% analysis_word & n > 30) %>%
  # Creates data for network graph
  graph_from_data_frame()

arrow_format <- grid::arrow(type = "closed", length = unit(.1, "inches"))

## Visual representation of connection of pair of words
ggraph(bg_grapgh, layout = "fr") +
  # Connection between words are represented by arrows
  geom_edge_link(aes(edge_alpha = n), 
                 show.legend = TRUE,
                 arrow = arrow_format, 
                 end_cap = circle(.1, 'inches')) +
  # Nodes for words
  geom_node_point(color = 'darkseagreen', 
                  size = 6) +
  # Text is displayed
  geom_node_text(aes(label = name), 
                 vjust = 1, 
                 hjust = 1,
                 repel = TRUE) +
  theme_void()

######
###### MAP EMOTIONS FOR EACH RATING ######
#########################################

# Creates unigrams from text
unigrams <- vegasrest %>% unnest_tokens(word, text, token = "ngrams", n = 1)
# nrc lexicon is loaded 
nrc <- get_sentiments("nrc")

sentiment_analysis <- unigrams %>% 
  dplyr::group_by(stars.x, word) %>% 
  # Count of words in review for each rating is calculated
  summarise( n = n()) %>% 
  # NRC sentiment analysis
  inner_join(nrc)

# positive and negative emotions are dropped
review_nrc <- sentiment_analysis %>%
  filter(!grepl("positive|negative", sentiment))


review_tally <- review_nrc %>%
  group_by(stars.x, sentiment) %>%
  tally() %>% 
  # Calculates the percentage of words that attribute to a sentiment
  mutate(cuisine_words = (nn / sum(nn))*100) %>% 
  select(-nn)

# Key value pairs
scores <- review_tally %>%
  spread(stars.x, cuisine_words)

# Plot radar chart
chartJSRadar(scores)

###########################################
### 4. DICTIONARY-BASED LOOKUP APPROACH ###
###########################################

######
###### Bing Dictionary ####
##########################

# word counting

vegasrestWords <- vegasrest %>%
  select(date,text,stars.x) %>%
  unnest_tokens("word", text)

vegasrestWords = vegasrestWords[-grep("1|2|3|4|5|6|7|8|9|0|yelp|the|a|an|in|food",
                                      vegasrestWords$word),]

# top 20 words after stop words

data("stop_words")

vegasrest_top_words<-
  vegasrestWords %>%
  anti_join(stop_words) %>%
  count(word) %>%
  arrange(desc(n))

top_20 <- vegasrest_top_words[1:20,]

#create factor variable to sort by frequency
vegasrest_top_words$word <- factor(vegasrest_top_words$word, levels = vegasrest_top_words$word[order(vegasrest_top_words$n,decreasing=TRUE)])

library(ggplot2)
ggplot(top_20, aes(x=word, y=n, fill=word))+
  geom_bar(stat="identity", fill = "lightpink1")+
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))+
  ylab("Most popular terms in Las Vegas Restaurant Reviews")+
  xlab("")+
  guides(fill=FALSE)

#tf-idf - Let's see some unusual words that appear, could be interesting?

vegasrest_tfidf<- vegasrest %>%
  select(date,text) %>%
  unnest_tokens("word", text) %>%
  anti_join(stop_words) %>%
  count(word, date) %>%
  bind_tf_idf(word, date, n)

top_tfidf <- vegasrest_tfidf %>%
  arrange(desc(tf_idf))

top_tfidf$word[1]

#Sentiment of reviews based on bing dictionary

vegasrest_sentiment <- vegasrestWords %>%
  left_join(get_sentiments("bing")) %>%
  count(date, word, stars.x, sentiment) 

#View sentiment of each start level (1-5 stars) based on word analysis (positive/negaive)

vegasrest_sentiment_plotstar <-
  vegasrestWords %>%
  inner_join(get_sentiments("bing")) %>% 
  count(stars.x, sentiment)

ggplot(vegasrest_sentiment_plotstar, aes(factor(stars.x), n, fill = sentiment)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Paired")

########
####### SentimentR Approach ###
##############################

load(file = "vegasrest.Rda")

sentiment=sentiment_by(vegasrest$text)
vegasrest$ave_sentiment<-sentiment$ave_sentiment
vegasrest$sd_sentiment<-sentiment$sd

####testing graphs (per each level of star rating 1-5)
review_1 <- vegasrest %>%  filter(stars.x >0 & stars.x <=1 )
review_2 <- vegasrest %>%  filter(stars.x >1 & stars.x <=2 )
review_3 <- vegasrest %>%  filter(stars.x >2 & stars.x <=3 )
review_4 <- vegasrest %>%  filter(stars.x >3 & stars.x <=4 )
review_5 <- vegasrest %>%  filter(stars.x >4 & stars.x <=5 )

qplot(review_1$ave_sentiment,   geom="histogram",binwidth=0.1,main="Review Sentiment Histogram")

summary(review_1$ave_sentiment)
summary(review_2$ave_sentiment)
summary(review_3$ave_sentiment)
summary(review_4$ave_sentiment)
summary(review_5$ave_sentiment)

########################################################
####### 5. MACHINE LEARNING SENTIMENT ANALYSIS #########
#######################################################


######
## Machine Learning Approach 1
###############################

# Define reviews that are clearly negative
vegasrest$Negative <- as.factor(vegasrest$stars.x <= 2)

# Define reviews that are clearly positive
vegasrest$Positive <- as.factor(vegasrest$stars.x>=4)

# Create Corpus
corpus <- VCorpus(VectorSource(vegasrest$text))

# Transform text to lowercase
corpus <- tm_map(corpus, tolower)

# converts corpus to a Plain Text Document
corpus <- tm_map(corpus, PlainTextDocument)

# Remove punctuation, stripwhitespace and numbers
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, stripWhitespace)

# Remove english stop words
corpus <- tm_map(corpus, removeWords, c("yelp", stopwords("english")))

# Stemming
corpus <- tm_map(corpus, stemDocument)

# DTM matrix
DTM <- DocumentTermMatrix(corpus)

# find most frequent terms
freq <- findFreqTerms(DTM, lowfreq = 50)

# remove sparce terms
sparse_DTM <- removeSparseTerms(DTM, 0.9)  ### 0.9 means to only keep terms that appear in 10% or more of the reviews, we took this threshold ###
### in order to have less variables for running the models, otherwise it takes too long to run ###

# Convert DTM to dataframe
vegasSparse <- as.data.frame(as.matrix(sparse_DTM))

# Fix variable names
colnames(vegasSparse) <- make.names(colnames(vegasSparse))

# Add dependent variables
vegasSparse$Negative <- vegasrest$Negative
vegasSparse$Positive <- vegasrest$Positive

# Split data in training and test
set.seed(123)

# split negative
splitNegative <- sample.split(vegasSparse$Negative, SplitRatio = 0.7)
trainSparseNegative <- subset(vegasSparse, splitNegative == TRUE)
testSparseNegative <- subset(vegasSparse, splitNegative == FALSE)

# split positive
splitPositive <- sample.split(vegasSparse$Positive, SplitRatio = 0.7)
trainSparsePositive <- subset(vegasSparse, splitPositive == TRUE)
testSparsePositive <- subset(vegasSparse, splitPositive == FALSE)

############ Decision Trees Model ################

# Predict sentiment for negative
vegasPredNegative <- rpart(Negative ~ . , data = trainSparseNegative, method = "class")
prp(vegasPredNegative)

# Predict sentiment for positive
vegasPredPositive <- rpart(Positive ~ . , data = trainSparsePositive, method = "class")
prp(vegasPredPositive)

# Prediction for negative sentiment
predictNegative <- predict(vegasPredNegative, newdata = testSparseNegative, type = "class")

# Prediction for positive sentiment
predictPositive <- predict(vegasPredPositive, newdata = testSparsePositive, type = "class")

# Confusion Matrix negative
cm_predNegative <- table(testSparseNegative$Negative, predictNegative)
cm_predNegative 

# Accuracy
accu_predNeg <- (cm_predNegative[1,1] + cm_predNegative[2,2])/sum(cm_predNegative)
accu_predNeg

# Confusion Matrix positive
cm_predPositive <- table(testSparsePositive$Positive, predictPositive)
cm_predPositive 

# Accuracy
accu_predPos <- (cm_predPositive[1,1] + cm_predPositive[2,2])/sum(cm_predPositive)
accu_predPos

###### Baseline model ########
cmat_baseline <- table(testSparseNegative$Negative)
cmat_baseline

accu_baseline <- max(cmat_baseline)/sum(cmat_baseline)
accu_baseline

cmat_baselineP <- table(testSparsePositive$Positive)
cmat_baselineP

accu_baselineP <- max(cmat_baselineP)/sum(cmat_baselineP)
accu_baselineP

######### Random Forest ##########
# Negative
set.seed(123)
vegasRFN <- randomForest(Negative ~ . , data = trainSparseNegative)
vegasRFN

# Positive
set.seed(123)
vegasRFP <- randomForest(Positive ~ . , data = trainSparsePositive)
vegasRFP

# Prediction and Accuracy
predictRFN <- predict(vegasRFN, newdata = testSparseNegative)
predictRFP <- predict(vegasRFP, newdata = testSparsePositive)

cmat_RFN <- table(testSparseNegative$Negative, predictRFN)
cmat_RFN

accu_RFN <- (cmat_RFN[1,1] + cmat_RFN[2,2])/sum(cmat_RFN)
accu_RFN

cmat_RFP <- table(testSparsePositive$Positive, predictRFP)
cmat_RFP

accu_RFP <- (cmat_RFP[1,1] + cmat_RFP[2,2])/sum(cmat_RFP)
accu_RFP

######## Logistic Regression Model #########
vegasLogN <- glm(Negative ~ . , data = trainSparseNegative, family = "binomial")
vegasLogN

vegasLogP <- glm(Positive ~ . , data = trainSparsePositive, family = "binomial")
vegasLogP

# Prediction and accuracy
predictLogN <- predict(vegasLogN, type = "response", newdata = testSparseNegative)
predictLogP <- predict(vegasLogP, type = "response", newdata = testSparsePositive)

cmat_LogN <- table(testSparseNegative$Negative, predictLogN > 0.5)
cmat_LogN

accu_LogN <- (cmat_LogN[1,1] + cmat_LogN[2,2])/sum(cmat_LogN)
accu_LogN

cmat_LogP <- table(testSparsePositive$Positive, predictLogP > 0.5)
cmat_LogP

accu_LogP <- (cmat_LogP[1,1] + cmat_LogP[2,2])/sum(cmat_LogP)
accu_LogP

##### ROC and AUC 

testSparsePositive$Positive1 <- ifelse(testSparsePositive$Positive == TRUE, 1, 0)
trainSparsePositive$Positive1 <- ifelse(trainSparsePositive$Positive == TRUE, 1, 0)
testSparseNegative$Negative1 <- ifelse(testSparseNegative$Negative == TRUE, 1, 0)
trainSparseNegative$Negative1 <- ifelse(trainSparseNegative$Negative == TRUE, 1, 0)

trainSparsePositive1 <- trainSparsePositive[, -c(174,175)]
testSparsePositive1 <- testSparsePositive[, -c(174,175)]
trainSparseNegative1 <- trainSparseNegative[, -c(174,175)]
testSparseNegative1 <- testSparseNegative[, -c(174,175)]

vegasLogN1 <- glm(Negative1 ~ . , data = trainSparseNegative1, family = "binomial")
vegasLogN1

vegasLogP1 <- glm(Positive1 ~ . , data = trainSparsePositive1, family = "binomial")
vegasLogP1

# Prediction and accuracy
predictLogN1 <- predict(vegasLogN1, type = "response", newdata = testSparseNegative1)
predictLogP1 <- predict(vegasLogP1, type = "response", newdata = testSparsePositive1)

# Calculate auc

#for negative
predLogN1 <- prediction(predictLogN1,testSparseNegative1$Negative1)
perfLogN1 <- performance(predLogN1,"tpr","fpr")
auc.perfLogN1 = performance(predLogN1, measure = "auc")
auc.perfLogN1@y.values

#for positive
predLogP1 <- prediction(predictLogP1,testSparsePositive1$Positive1)
perfLogP1 <- performance(predLogP1,"tpr","fpr")
auc.perfLogP1 = performance(predLogP1, measure = "auc")
auc.perfLogP1@y.values

# ROC curve
perfDTneg1 <- performance(predDTpos1,"tpr","fpr")
plot(perfDTneg1)
abline(0,1)

#Plot AUC Curve based on the variables
plot(perfLogN1, col="red", type = "l", xlab="variables",ylab="auc value",
     main = "ROC Curves Logistic Regression", lwd = 2)
par(new=TRUE)
plot(perfLogP1,col="blue", type = "l", lwd = 2)

legend("bottom", legend=c("negative","positive"), ncol=2,bty="n",
       col=c("red", "blue"), lwd = 2)
abline(0,1)
grid (NULL,NULL, lty = 6)

######
## Machine Learning Approach 2
################################

# 1. create a complete data set

load(file = "vegasrest.Rda")
vegasrest$sentiment <- ifelse(vegasrest$stars.x > 4, 1,0)
head(vegasrest)

# 2. Create a training and test set
SentimentReal<-vegasrest
set.seed(2) # Set a seed to have the same subsets every time 

y <- as.factor(SentimentReal[,"sentiment"])
levels(y)

# Define proportion to be in training set 
p <- 0.8

# Define observations to be in training set
class1_train <- sample(which(y==as.integer(levels(y)[1])), floor(p*table(y)[1]),replace=FALSE)
class2_train <- sample(which(y==as.integer(levels(y)[2])), floor(p*table(y)[2]),replace=FALSE)

training_locations <- c(class1_train,class2_train) 

# Create the training and test set; Store them in a list for convenience

txt_l <- list()
txt_l[[2]] <- list()

txt_l[[1]]<- SentimentReal[sort(training_locations),3]
txt_l[[2]]<- SentimentReal[-sort(training_locations),3]


# Make the training and test set corpora
for (i in 1:2){
  txt_l[[i]] <- VCorpus(VectorSource((txt_l[[i]])))
}

# 3. Create the term-document matrices
# create a function that allows to make the training and test set correctly, with the n-grams specified
Ngram <- function(inputset1,inputset2,mindegree,maxdegree){
  # inputset1 = training dataset
  # inputset2 = test dataset
  # mindegree = minimum n-gram
  # maxdegree = maximum n-gram
  
  outputlist <- list()
  
  # training
  Tokenizer <- function(x) NGramTokenizer(x, Weka_control(min = mindegree, max = maxdegree))
  tr <- DocumentTermMatrix(inputset1, control = list(tokenize = Tokenizer,
                                                     weighting = function(x) weightTf(x),
                                                     RemoveNumbers=TRUE,
                                                     removePunctuation=TRUE,
                                                     stripWhitespace= TRUE))
  # test
  test <- DocumentTermMatrix(inputset2, control = list(tokenize = Tokenizer,
                                                       weighting = function(x) weightTf(x),
                                                       RemoveNumbers=TRUE,
                                                       removePunctuation=TRUE,
                                                       stripWhitespace= TRUE))
  
  # Reform the test DTM to have the same terms as the training case 
  Intersect <- test[,intersect(colnames(test), colnames(tr))]
  diffCol <- tr[,setdiff(colnames(tr),colnames(test))]
  newCols <- as.simple_triplet_matrix(matrix(0,nrow=test$nrow,ncol=diffCol$ncol))
  newCols$dimnames <- diffCol$dimnames
  testNew<-cbind(Intersect,newCols)
  testNew<- testNew[,colnames(tr)]
  
  ## Convert term document matrices to common sparse matrices to apply efficient SVD algorithm
  
  dtm.to.sm <- function(dtm) {sparseMatrix(i=dtm$i, j=dtm$j, x=dtm$v,dims=c(dtm$nrow, dtm$ncol))}
  
  outputlist<- list(train=dtm.to.sm(tr),test=dtm.to.sm(testNew))
  
  return(outputlist)
}

# apply our function, stored in a new list, unigram
require(slam)
library(slam)

unigram <-Ngram(txt_l[[1]],txt_l[[2]],1,1)

# 4. Apply Singular Value Decomposition

SVD_all <- function(inputset,k){
  outputlist <- list()
  
  outputlist[[i]]<-list()
  
  trainer <- irlba(t(inputset[[1]]), nu=k, nv=k)
  tester <- as.data.frame(as.matrix(inputset[[2]] %*% trainer$u %*%  solve(diag(trainer$d))))
  
  outputlist<- list(train = as.data.frame(trainer$v), test= tester)
  
  return(outputlist)
}

svdUnigram <- SVD_all(unigram,20)

# 5. Prediction models

# Create datasets to use: append our dependent variable to our dataset 

train  <- cbind(y[sort(training_locations)],svdUnigram[[1]])
test <- cbind(y[-sort(training_locations)],svdUnigram[[2]])

## Apply Random Forest

RF_model_train <- randomForest(x=train[,2:dim(train)[[2]]],y=train[,1],importance=TRUE,ntree=1001)
RF_predict <- predict(RF_model_train,test[,2:dim(test)[[2]]],type = "prob")[,2]

# This returns the probabilities, which is more useful for the evaluation measures

RF_model_train <- randomForest(x=train[,2:dim(train)[[2]]],y=train[,1],importance=TRUE,ntree=1001)
RF_predict <- predict(RF_model_train,test[,2:dim(test)[[2]]],type = "prob")[,2]

# Calculate auc

predML <- prediction(RF_predict,test[,1])

# ROC curve
perfML <- performance(predML,"tpr","fpr")
plot(perfML)
abline(0,1)

## auc
auc.perfML = performance(predML, measure = "auc")
auc.perfML@y.values


