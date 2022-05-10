from abc import ABC, abstractmethod
from gensim import models
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, AutoModel

import fasttext
import torch
import numpy as np

class VectorSpaceModel(ABC):
    """
    Abstract class to represent a vector space model
    
    Methods
    -------
    create_embedding(lst_token, per_word)
        Create word embedding from words in 'lst_token' using 'per_word' method
    """
    
    @abstractmethod
    def create_embedding(self, lst_token, per_word):
        """Create word embedding from words in 'lst_token' using 'per_word' method

        Parameters
        ----------
        lst_token: list of string
            list of token that will be converted
        per_word: boolean 
            embedding method,
            True: embedding value will be computed on each token
            False: embedding value will be computed on each sentence 

        Returns
        -------
        numpy array
            vector representation of lst_token using a vector space model 
        """
        
        pass

class TrainedModel(ABC):
    """
    Abstract class to represent a vector space model that have to be trained before using it
    
    Methods
    -------
    train(doc)
        train the vector space model using "doc" corpus
    """
    
    @abstractmethod
    def train(self, doc):
        """Train the vector space model using "doc" corpus

        Parameters
        ----------
        doc: two dimensional list
            every index is a list of string that be used for the training process

        Returns
        -------
        None
        """
    
        pass
    
class PreTrainedModel(ABC):
    """
    Abstract class to represent a vector space model that can use a pretrained model
    """
    
    pass

class BoWModel(VectorSpaceModel, TrainedModel):
    """
    Concrete class to represent a vector space model using Bag-of-Words
    
    Attributes
    ----------
    vectorizer : CountVectorizer()
        Bag-of-Words model 
    """
    
    vectorizer = None

    def train(self, doc):
        data = [" ".join(sent) for sent in doc]
        model = CountVectorizer()
        self.vectorizer = model.fit(data)
    
    def create_embedding(self, lst_token, per_word):
        sent = [" ".join(lst_token)]
        res = self.vectorizer.transform(sent)
        return res.toarray()[0]

class TfIdfModel(VectorSpaceModel, TrainedModel):
    """
    Concrete class to represent a vector space model using TF-IDF
    
    Attributes
    ----------
    vectorizer : TfidfVectorizer()
        TF-IDF model 
    """
    
    vectorizer = None

    def train(self, doc):
        data = [" ".join(sent) for sent in doc]
        model = TfidfVectorizer()
        self.vectorizer = model.fit(data)
    
    def create_embedding(self, lst_token, per_word):
        sent = [" ".join(lst_token)]
        res = self.vectorizer.transform(sent)
        return res.toarray()[0]

class Word2VecModel(VectorSpaceModel, PreTrainedModel):
    """
    Concrete class to represent a vector space model using Word2Vec
    
    Attributes
    ----------
    default_model : string
        default pretrained model file path relative to this file location 
    """
    
    default_model = "../pretrained_model/word2vec.id.300d.txt.gz"
    
    def __init__(self, params):
        pretrained_file = params['pretrained_file'] if 'pretrained_file' in params.keys() else self.default_model
        self.model = models.KeyedVectors.load_word2vec_format(pretrained_file)
    
    def create_embedding(self, lst_token, per_word):
        rep = []
        for token in lst_token:
            try:
                token_rep = self.model[token]
                rep.append(token_rep)
            except:
                continue
        return np.mean(rep, axis=0)

class FastTextModel(VectorSpaceModel, PreTrainedModel):
    """
    Concrete class to represent a vector space model using FastText
    
    Attributes
    ----------
    default_model : string
        default pretrained model file path relative to this file location 
    """

    default_model = "../pretrained_model/cc.id.300.bin"

    def __init__(self, params):
        pretrained_file = params['pretrained_file'] if 'pretrained_file' in params.keys() else self.default_model
        self.model = fasttext.load_model(pretrained_file) 
    
    def create_embedding(self, lst_token, per_word):
        rep = []
        for token in lst_token:
            try:
                token_rep = self.model[token]
                rep.append(token_rep)
            except:
                continue
        return np.mean(rep, axis=0)

class BERTModel(VectorSpaceModel, PreTrainedModel):
    """
    Concrete class to represent a vector space model using BERT
    
    Attributes
    ----------
    default_batch_size: int
        default batch_size value for create vector embedding
    default_model : string
        default pretrained model file path relative to this file location 
    default_device : string
        default gpu device name
    """
    
    default_batch_size = 8 
    default_model = "indolem/indobert-base-uncased"
    default_device = "cuda:0"
    
    def __init__(self, params):
        self.batch_size = params['batch_size'] if 'batch_size' in params.keys() else self.default_batch_size
        if torch.cuda.is_available:
            self.device = params['device'] if 'device' in params.keys() else self.default_device
        else:
            self.device = "cpu"
        pretrained_file = params['pretrained_file'] if 'pretrained_file' in params.keys() else self.default_model
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_file)
        self.model =  AutoModel.from_pretrained(pretrained_file, output_hidden_states = True).to(self.device)
    
    def create_embedding(self, doc, per_word):
        if per_word:
            return self.create_topic_embedding(doc)
        else:
            return self.create_sentence_embedding(doc)
    
    def create_topic_embedding(self, lst_topic):
        """Create word embedding from words in lst_topic

        Parameters
        ----------
        lst_topic: two dimensional list
            each index represent list of word that describe a topic 
            
        Returns
        -------
        list
            list that contains vector representation of each topic in numpy array form
        """
        
        rep = []
        for topic_vec in self.batch(lst_topic):
            input_dict = self.get_input_dict(topic_vec)
            rep += self.get_embedding(input_dict)
        return rep

    def create_sentence_embedding(self, document):
        """Create word embedding from sentences in document

        Parameters
        ----------
        document: two dimensional list
            each index represent list of word on a sentence 
            
        Returns
        -------
        list
            list that contains vector representation of each sentence in numpy array form
        """
        
        cleaned_doc = self.preprocess(document)
        rep = []
        for lst_doc in self.batch(cleaned_doc):
            input_dict = self.get_input_dict(lst_doc)
            rep += self.get_embedding(input_dict)
        return rep
    
    def preprocess(self, document):
        """Convert each list-of-word in document into sentence 

        Parameters
        ----------
        document: two dimensional list
            each index represent list of word on a sentence  
            
        Returns
        -------
        list
            list that contain all sentences on document
        """
        
        cleaned_doc = []
        num_of_sent = len(document)
        for i in range(num_of_sent):
            sent = document[i]
            sentence = " ".join(sent)
            cleaned_doc.append(sentence) 
        return cleaned_doc

    def get_input_dict(self, document):
        """Convert each sentence in document into dictionary that can be used for BERT Model

        Parameters
        ----------
        document: list 
            list that contain several sentence 
            
        Returns
        -------
        dict
            dictionary that can be used for BERT Model
        """
        
        lst_input_chunks = []
        lst_mask_chunks = []
        for sent in document:
            tokens = self.tokenizer.encode_plus(sent, add_special_tokens=True, return_tensors='pt')
            input_chunk = tokens['input_ids'][0]
            mask_chunk = tokens['attention_mask'][0]
            pad_len = 512 - input_chunk.shape[0]
            if pad_len > 0:
                input_chunk = torch.cat([
                    input_chunk, torch.Tensor([0] * pad_len)
                ])
                mask_chunk = torch.cat([
                    mask_chunk, torch.Tensor([0] * pad_len)
                ])
            lst_input_chunks.append(input_chunk)
            lst_mask_chunks.append(mask_chunk)
        input_ids = torch.stack(lst_input_chunks).to(self.device)
        attention_mask = torch.stack(lst_mask_chunks).to(self.device)
        input_dict = {
                    'input_ids': input_ids.long(),
                    'attention_mask': attention_mask.int()
        }
        return input_dict
    
    def get_embedding(self, input_dict):
        """Convert sentences that represented by input_dict into its BERT vector embedding
        using sum of 4 last BERT layer

        Parameters
        ----------
        input_dict: dict 
            sentences that being converted in the form of BERT model dict input
            
        Returns
        -------
        list
            list of vector embeddings, 
            each index represent an embeddings for a sentence in the same index on the input
        """
        
        lst_vec = self.model(**input_dict)['hidden_states']
        lst_embedding = lst_vec[-4:]
        num_of_part = lst_embedding[0].shape[0]
        fin_embedding = []
        for i in range(num_of_part):
            tmp = []
            for embedding in lst_embedding:
                tmp2 = embedding[i]
                tmp.append(torch.mean(tmp2, dim=0).cpu().detach().numpy())
            fin_embedding.append(np.sum(tmp, axis=0))
        return fin_embedding
    
    def batch(self, iterable):
        """Divide an iterable object into several parts that has size <= self.batch_size

        Parameters
        ----------
        iterable: list 
            list of sentences 
            
        Returns
        -------
        list
        """
        
        l = len(iterable)
        for ndx in range(0, l, self.batch_size):
            yield iterable[ndx:min(ndx + self.batch_size, l)]
