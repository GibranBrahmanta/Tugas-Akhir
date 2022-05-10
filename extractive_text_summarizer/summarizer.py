from .similarity_measure import *
from .topic_modeling import *
from .vector_space_model import *
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Summarizer:
    """
    Concrete class to represent an extractive text summarizer that use Belwal et al. (2021) method
    
    Attributes
    ----------
    stop_factory: StopWordRemoverFactory()
        Stopword Remover
    lst_stopword: list
        list of stopword that will be used
    factory: Stemmer Factory()
        object to create stemmer
    stemmer: Stemmer()
        stemmer
    
    Methods
    -------
    create_topic_model(topic_modeling)
        init used topic_model based on 'topic_modeling'
    create_vsm_model(vsm_model)
        init used vector space model based on 'vsm_model'
    create_similarity_metric(similarity)
        init used similiarity metric based on 'similarity'
    preprocess(document)
        remove stopword, punctuation, number and lower all characters in document
    create_combined_topic(lst_topic_word)
        merge all topic representation in 'lst_topic_word' into a list
    create_topic_word(doc, num_of_topic, num_of_words, ranking_method)
        create topic vector related to 'doc' based on several details that described by other parameters
    create_embedding(doc, per_word)
        create word embedding from words in 'doc' using per_word method
    compute_similarity(v1, v2)
        calculate similarity between v1 and v2 
    get_sum_index(doc_vec, topic_vec, num_of_sentence)
        get summary index based on 'doc_vec' and 'topic_vec' 
    get_sum(doc, lst_sum_idx)
        get summary sentences based on 'lst_sum_idx'
    train_vsm(document)
        train used vsm model using 'document' corpus
    summarize(document, num_of_topic, num_of_words, ranking_method)
        summarize 'document' based on several details that described by other parameters
    """
    
    stop_factory = StopWordRemoverFactory()
    lst_stopword = stop_factory.get_stop_words()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def __init__(self, topic_modeling, vsm_model, similarity):
        self.topic_model = self.create_topic_model(topic_modeling)
        self.vsm_model = self.create_vsm_model(vsm_model)
        self.similarity_metric = self.create_similarity_metric(similarity)

    def create_topic_model(self, topic_modeling):
        """Init used topic modeling method based on 'topic_modeling'

        Parameters
        ----------
        topic_modeling: string
            topic modeling method that will be used

        Returns
        -------
        TopicModeling()
        """
        
        model = None
        if topic_modeling == "LDA":
            model = LDAModel()
        elif topic_modeling == "LSA":
            model = LSAModel()
        elif topic_modeling == "NMF":
            model = NMFModel()
        return model
    
    def create_vsm_model(self, vsm_model):
        """Init used vsm based on 'vsm_model'

        Parameters
        ----------
        vsm: dict
            dict contains vsm data that will be used

        Returns
        -------
        VectorSpaceModel()
        """
        
        model = None
        model_name = vsm_model['model_name']
        if model_name == "Word2Vec":
            model = Word2VecModel(vsm_model)
        elif model_name == "BoW":
            model = BoWModel()
        elif model_name == "TF-IDF":
            model = TfIdfModel()
        elif model_name == "FastText":
            model = FastTextModel(vsm_model)
        elif model_name == "BERT":
            model = BERTModel(vsm_model)
        return model
    
    def create_similarity_metric(self, similarity):
        """Init used similarity metric based on 'similarity'

        Parameters
        ----------
        similarity: string
            similarity metric that will be used

        Returns
        -------
        SimilarityMeasure()
        """
        
        metric = None
        if similarity == "Cosine":
            metric = CosineSimilarity()
        elif similarity == "Euclidean":
            metric = EuclideanDistance()
        elif similarity == "Jaccard":
            metric = JaccardCoefficient()
        return metric

    def preprocess(self, document):
        """Remove stopword, punctuation, number and lower all characters in document

        Parameters
        ----------
        document: two dimentional list
            document that will be used for the summarization process

        Returns
        -------
        two dimensional list
            cleaned version of 'document'
        """
        
        cleaned_doc = []
        for sent in document:
            tmp = []
            for token in sent:
                if re.match("[a-zA-Z]+", token) and token not in self.lst_stopword:
                    new_token = token.lower()
                    new_token = self.stemmer.stem(new_token)
                    tmp.append(new_token)
            cleaned_doc.append(tmp)                
        return cleaned_doc
    
    def create_combined_topic(self, lst_topic_word):
        """Merge all topic representation in 'lst_topic_word' into a list

        Parameters
        ----------
        lst_topic_word: two dimentional list
            each index contains representation of a topic

        Returns
        -------
        list
            combined topic vector
        """
        
        res = []
        for topic in lst_topic_word:
            for word in topic:
                res.append(word)
        return [res]
    
    def create_topic_word(self, doc, num_of_topic, num_of_words, ranking_method):
        """Create topic vector related to 'doc' based on 
        several details that described by other parameters

        Parameters
        ----------
        doc: two dimentional list
            document that will be used for the summarization process
        num_of_topic: int
            the number of searched topic
        num_of_words: int
            the number of words that used to represent a topic
        ranking_method: string (Individual/Combined)
            ranking method that will be used to generate summary

        Returns
        -------
        two dimensional list
            each index contains representation of a/all topic
        """
        
        lst_topic_word = self.topic_model.get_topic_word(doc, num_of_topic, num_of_words)
        if ranking_method == "Combined":
            lst_topic_word = self.create_combined_topic(lst_topic_word)
        return lst_topic_word
    
    def create_embedding(self, doc, per_word):
        """Create word embedding from words in 'doc' using 'per_word' method 
        (if Jaccard Coefficient not used in the similarity metric)

        Parameters
        ----------
        doc: two dimesional list
            document that will be used for the summarization process
        per_word: boolean 
            embedding method,
            True: embedding value will be computed on each token
            False: embedding value will be computed on each sentence 

        Returns
        -------
        numpy array
            vector representation of lst_token using a vector space model
        or
        two dimensional list
            'doc' parameter
        """
        
        if type(self.similarity_metric) == JaccardCoefficient:
            return doc
        if isinstance(self.vsm_model, BERTModel):
            return self.vsm_model.create_embedding(doc, per_word)
        else:
            res = []
            for i in doc:
                res.append(self.vsm_model.create_embedding(i, per_word))
            return res
    
    def compute_similarity(self, v1, v2):
        """Calculate similarity between v1 and v2 using a similarity metrics

        Parameters
        ----------
        v1: numpy array
            one dimensional vector
        v2: numpy array 
            one dimensional 

        Returns
        -------
        float
            similarity value between v1 and v2. 
        """
        
        return self.similarity_metric.compute_similarity(v1, v2)
    
    def get_sum_index(self, doc_vec, topic_vec, num_of_sentence):
        """Get summary index based on 'doc_vec' and 'topic_vec' 

        Parameters
        ----------
        doc_vec: two dimentional list
            list of vector representation of each sentence in input doc
        topic_vec: two dimentional list
            list of vector representation of each topic representation in input doc
        num_of_sentence: int
            the number of sentence that will be included in the summary result

        Returns
        -------
        list
            list of summary index
        """
        
        res = []
        if len(topic_vec) == 1:
            tmp = []
            for i in range(len(doc_vec)):
                if type(doc_vec[i]) == np.float64:
                    continue
                sim = self.compute_similarity([doc_vec[i]], topic_vec)
                tmp.append([i, sim])
            tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
            for i in range(num_of_sentence):
                res.append(tmp[i][0])
        else:
            tmp = []
            skip_idx = []
            for i in range(len(topic_vec)):
                tmp2 = []
                for j in range(len(doc_vec)):
                    if type(doc_vec[j]) == np.float64 or type(topic_vec[i]) == np.float64:
                        if type(topic_vec[i]) == np.float64:
                            skip_idx.append(i)
                        continue
                    sim = self.compute_similarity([doc_vec[j]], [topic_vec[i]])
                    tmp2.append([j, sim])
                tmp2 = sorted(tmp2, key=lambda x: x[1], reverse=True)
                tmp.append(tmp2)
            cnt = 0
            while cnt < num_of_sentence:
                if cnt not in skip_idx:
                    for i in range(len(tmp[cnt])):
                        idx = tmp[cnt][i][0]
                        if idx not in res:
                            res.append(idx)
                            break
                cnt += 1
        return res
    
    def get_sum(self, doc, lst_sum_idx):
        """Get summary sentences based on 'lst_sum_idx'

        Parameters
        ----------
        doc: two dimentional list
            document that will be used for the summarization process
        lst_sum_idx: list
            list of summary index

        Returns
        -------
        list
            list of summary sentences
        """
        
        res = []
        lst_idx = sorted(lst_sum_idx)
        for idx in lst_idx:
            res.append(doc[idx])
        return res
    
    def train_vsm(self, document):
        """Train used vsm model using 'document' corpus

        Parameters
        ----------
        document: two dimentional list
            document that will be used for the training process

        Returns
        -------
        None
        """
        
        self.vsm_model.train(document)
    
    def summarize(self, document, num_of_topic, num_of_words, ranking_method):
        """summarize 'document' based on several details that described by other parameters

        Parameters
        ----------
        document: two dimentional list
            document that will be used for the summarization process
        num_of_topic: int
            the number of searched topic
        num_of_words: int
            the number of words that used to represent a topic
        ranking_method: string (Individual/Combined)
            ranking method that will be used to generate summary

        Returns
        -------
        list
            list of summary sentences
        list
            list of used topic word for generating the summary
        """
        
        cleaned_doc = self.preprocess(document)
        if isinstance(self.vsm_model, TrainedModel):
            self.train_vsm(cleaned_doc)
        lst_topic_word = self.create_topic_word(cleaned_doc, num_of_topic, num_of_words, ranking_method)
        doc_vec = self.create_embedding(cleaned_doc, False)
        topic_vec = self.create_embedding(lst_topic_word, True)
        lst_sum_idx = self.get_sum_index(doc_vec, topic_vec, num_of_topic)
        return self.get_sum(document, lst_sum_idx), lst_topic_word
