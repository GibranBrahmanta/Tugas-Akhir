from abc import ABC, abstractmethod
from gensim import corpora, models
import re

class TopicModeling(ABC):
    """
    Abstract class to represent a topic modeling method
    
    Methods
    -------
    get_topic_word(doc, num_of_topic, num_of_words)
        get 'num_of_topic' topics from 'doc' where every topic is represented by 'num_of_words' words
    """
    
    @abstractmethod
    def get_topic_word(self, doc, num_of_topic, num_of_words):
        """get 'num_of_topic' topics from 'doc' where every topic is represented by 'num_of_words' words

        Parameters
        ----------
        doc: two dimensional list
            every index is a list of string 
        num_of_topic: int
            the number of searched topic
        num_of_words: int
            the number of words that used to represent a topic

        Returns
        -------
        two dimesional list
            each index represent a topic using several words
        """
    
        pass

class LDAModel(TopicModeling):
    """
    Concrete class to represent a topic modeling using Latent Dirichlet Allocation
    
    Methods
    -------
    parse_topic(query, num_of_words)
        parse topic data represented by 'query' into list 
    sort_topic(corpus, lst_topic, lda_model)
        sort topic on 'corpus' that given by 'lda_model' based on its occurence (descending)
    get_highest_topic(lst_topic)
        get topic with highest probability in 'lst_topic'
    """
    
    def parse_topic(self, query, num_of_words):
        """Parse topic data represented by 'query' into list

        Parameters
        ----------
        query: string
            string representation of a topic data
        num_of_words:
            the number of words that used to represent a topic 
            
        Returns
        -------
        list
            list that contains words that represent a topic
        """
        
        pattern = "[a-zA-Z]+"
        tmp_res = re.findall(pattern, query)
        return tmp_res[:num_of_words]
    
    def get_topic_word(self, doc, num_of_topic, num_of_words):
        id2word = corpora.Dictionary(doc)
        corpus = [id2word.doc2bow(text) for text in doc]
        lda_model = models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_of_topic, 
                                            random_state=2022,
                                            per_word_topics=True)
        topic_vec = []
        lst_topic = lda_model.print_topics()
        ordered_topic = self.sort_topic(corpus, lst_topic, lda_model)
        for index in ordered_topic:
            topic = lst_topic[index]
            topic_vec.append(self.parse_topic(topic[1], num_of_words))
        return topic_vec
    
    def sort_topic(self, corpus, lst_topic, lda_model):
        """Sort topic on 'corpus' that given by 'lda_model' based on its occurence (descending)

        Parameters
        ----------
        corpus: list 
            list representation of used documents
        lst_topic: list
            list of topic at 'corpus' that given by the 'lda_model' 
        lda_model: LdaModel()
            LDA model
            
        Returns
        -------
        list
            list that contains sorted topic by its occurence
        """
        
        ordered_topic = [[topic[0], 0] for topic in lst_topic]
        for doc in corpus:
            topic = lda_model.get_document_topics(doc)
            topic_idx = self.get_highest_topic(topic)
            for t in ordered_topic:
                if t[0] == topic_idx:
                    t[1] += 1
        ordered_topic = sorted(ordered_topic, key=lambda x: x[1], reverse=True)
        return [i[0] for i in ordered_topic]
    
    def get_highest_topic(self, lst_topic):
        """Get topic with highest probability in 'lst_topic'

        Parameters
        ----------
        lst_topic: list
            list of [topic, probability]
            
        Returns
        -------
        string
            string representation of a topic that has biggest probability
        """
        
        res = lst_topic[0][0]
        curr_prob = lst_topic[0][1]
        for i in range(1, len(lst_topic)):
            if lst_topic[i][1] > curr_prob:
                res = lst_topic[i][0]
                curr_prob = lst_topic[i][1]
        return res

class NMFModel(TopicModeling):
    """
    Concrete class to represent a topic modeling using Non-Negative Matrix Factorization
    
    Methods
    -------
    parse_topic(query, num_of_words)
        parse topic data represented by 'query' into list 
    sort_topic(corpus, lst_topic, nmf_model)
        order topic on 'corpus' that given by 'nmf_model' based on its occurence (descending)
    get_highest_topic(lst_topic)
        get topic with highest probability in 'lst_topic'
    """
    
    def parse_topic(self, query, num_of_words):
        """Parse topic data represented by 'query' into list

        Parameters
        ----------
        query: string
            string representation of a topic data
        num_of_words:
            the number of words that used to represent a topic 
            
        Returns
        -------
        list
            list that contains words that represent a topic
        """
        
        pattern = "[a-zA-Z]+"
        tmp_res = re.findall(pattern, query)
        return tmp_res[:num_of_words]
    
    def get_topic_word(self, doc, num_of_topic, num_of_words):
        id2word = corpora.Dictionary(doc)
        corpus = [id2word.doc2bow(text) for text in doc]
        nmf_model = models.nmf.Nmf(corpus=corpus,
                            id2word=id2word,
                            num_topics=num_of_topic, 
                            random_state=2022)
        topic_vec = []
        lst_topic = nmf_model.print_topics()
        ordered_topic = self.sort_topic(corpus, lst_topic, nmf_model)
        for index in ordered_topic:
            topic = lst_topic[index]
            topic_vec.append(self.parse_topic(topic[1], num_of_words))
        return topic_vec
    
    def sort_topic(self, corpus, lst_topic, nmf_model):
        """Sort topic on 'corpus' that given by 'nmf_model' based on its occurence (descending)

        Parameters
        ----------
        corpus: list 
            list representation of used documents
        lst_topic: list
            list of topic at 'corpus' that given by the 'nmf_model' 
        lda_model: Nmf()
            NMF model
            
        Returns
        -------
        list
            list that contains sorted topic by its occurence
        """
        
        ordered_topic = [[topic[0], 0] for topic in lst_topic]
        for doc in corpus:
            topic = nmf_model.get_document_topics(doc)
            if len(doc) != 0:
                topic_idx = self.get_highest_topic(topic)
            else:
                continue
            for t in ordered_topic:
                if t[0] == topic_idx:
                    t[1] += 1
        ordered_topic = sorted(ordered_topic, key=lambda x: x[1], reverse=True)
        return [i[0] for i in ordered_topic]
    
    def get_highest_topic(self, lst_topic):
        """Get topic with highest probability in 'lst_topic'

        Parameters
        ----------
        lst_topic: list
            list of [topic, probability]
            
        Returns
        -------
        string
            string representation of a topic that has biggest probability
        """
        
        res = lst_topic[0][0]
        curr_prob = lst_topic[0][1]
        for i in range(1, len(lst_topic)):
            if lst_topic[i][1] > curr_prob:
                res = lst_topic[i][0]
                curr_prob = lst_topic[i][1]
        return res

class LSAModel(TopicModeling):
    """
    Concrete class to represent a topic modeling using Latent Semantic Analysis
    
    Methods
    -------
    parse_topic(query, num_of_words)
        parse topic data represented by 'query' into list 
    sort_topic(corpus, lst_topic, lsa_model)
        order topic on 'corpus' that given by 'lsa_model' based on its occurence (descending)
    get_highest_topic(lst_topic)
        get topic with highest probability in 'lst_topic'
    """
    
    def parse_topic(self, query, num_of_words):
        """Parse topic data represented by 'query' into list

        Parameters
        ----------
        query: string
            string representation of a topic data
        num_of_words:
            the number of words that used to represent a topic 
            
        Returns
        -------
        list
            list that contains words that represent a topic
        """
        
        pattern = "[a-zA-Z]+"
        tmp_res = re.findall(pattern, query)
        return tmp_res[:num_of_words]
    
    def get_topic_word(self, doc, num_of_topic, num_of_words):
        id2word = corpora.Dictionary(doc)
        corpus = [id2word.doc2bow(text) for text in doc]
        lsa_model = models.lsimodel.LsiModel(corpus=corpus,
                            id2word=id2word,
                            num_topics=num_of_topic)
        topic_vec = []
        lst_topic = lsa_model.print_topics()
        ordered_topic = self.sort_topic(corpus, lst_topic, lsa_model)
        for index in ordered_topic:
            topic = lst_topic[index]
            topic_vec.append(self.parse_topic(topic[1], num_of_words))
        return topic_vec
    
    def sort_topic(self, corpus, lst_topic, lsa_model):
        """Sort topic on 'corpus' that given by 'lsa_model' based on its occurence (descending)

        Parameters
        ----------
        corpus: list 
            list representation of used documents
        lst_topic: list
            list of topic at 'corpus' that given by the 'lsa_model' 
        lda_model: LsiModel()
            LSA model
            
        Returns
        -------
        list
            list that contains sorted topic by its occurence
        """
        
        ordered_topic = [[topic[0], 0] for topic in lst_topic]
        lst_topic = lsa_model[corpus]
        for i in range(len(corpus)):
            topic = lst_topic[i]
            if len(topic) != 0:
                    topic_idx = self.get_highest_topic(topic)
            else:
                continue
            for t in ordered_topic:
                if t[0] == topic_idx:
                    t[1] += 1
        ordered_topic = sorted(ordered_topic, key=lambda x: x[1], reverse=True)
        return [i[0] for i in ordered_topic]
    
    def get_highest_topic(self, lst_topic):
        """Get topic with highest probability in 'lst_topic'

        Parameters
        ----------
        lst_topic: list
            list of [topic, probability]
            
        Returns
        -------
        string
            string representation of a topic that has biggest probability
        """
        
        res = lst_topic[0][0]
        curr_prob = lst_topic[0][1]
        for i in range(1, len(lst_topic)):
            if lst_topic[i][1] > curr_prob:
                res = lst_topic[i][0]
                curr_prob = lst_topic[i][1]
        return res