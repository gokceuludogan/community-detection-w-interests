import os
from collections import OrderedDict
import itertools
import abc
import re
try:
    import rdflib as rdf
except ImportError:
    pass

import networkx as nx
import numpy as np
from ontology import Ontology
import dgl
import dgl.backend as F
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info, _get_dgl_url
from dgl.data.utils import generate_mask_tensor, idx2mask, deprecate_property, deprecate_class

RENAME_DICT = {
    'type' : 'rdftype',
    'rev-type' : 'rev-rdftype',
}
class Entity:
    """Class for entities
    Parameters
    ----------
    id : str
        ID of this entity
    cls : str
        Type of this entity
    """
    def __init__(self, e_id, cls):
        self.id = e_id
        self.cls = cls

    def __str__(self):
        return '{}/{}'.format(self.cls, self.id)

class Relation:
    """Class for relations
    Parameters
    ----------
    cls : str
        Type of this relation
    """
    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return str(self.cls)

class RDFGraphDataset(DGLBuiltinDataset):
    #'../data/deneme/'
    def __init__(self, name=None, url=None,
                 raw_dir='../data/deneme/',
                 force_reload=False,
                 verbose=True):
        self._insert_reverse = True
        self._print_every = 10000
        #self.save_path = "../data/deneme/"
        super(RDFGraphDataset, self).__init__(name, url,
                                              raw_dir=raw_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)

    def process(self):
        raw_tuples = self.load_raw_tuples(self.raw_path)
        self.process_raw_tuples(raw_tuples)

    def load_raw_tuples(self, root_path):
        """Loading raw RDF dataset

        Parameters
        ----------
        root_path : str
            Root path containing the data

        Returns
        -------
            Loaded rdf data
        """
        raw_rdf_graphs = []
        for _, filename in enumerate([root_path]):
            """fmt = None
            if filename.endswith('nt'):
                fmt = 'nt'
            elif filename.endswith('n3'):
                fmt = 'n3'
            elif filename.endswith('rdf'):
                fmt = 'application/rdf+xml'
            if fmt is None:
                continue"""
            #g = rdf.Graph()
            print("Hadi bakalım")
            ontology = Ontology(filename)
            interests = ontology.get_all_interests()
            print("Geliyor")
            #print('Parsing file %s ...' % filename)
            """mentions = ontology.get_all_interactions('mention')
            retweets = ontology.get_all_interactions('retweet')
            concated = mentions + retweets"""
            #g.parse(filename, format=fmt)
            raw_rdf_graphs.append(interests)
            #raw_rdf_graphs.append(retweets)
        return itertools.chain(*raw_rdf_graphs)

    def process_raw_tuples(self, raw_tuples):
        """Processing raw RDF dataset

        Parameters
        ----------
        raw_tuples:
            Raw rdf tuples
        root_path: str
            Root path containing the data
        """
        mg = nx.MultiDiGraph()
        ent_classes = OrderedDict()
        rel_classes = OrderedDict()
        entities = OrderedDict()
        src = []
        dst = []
        ntid = []
        etid = []
        sorted_tuples = []
        for t in raw_tuples:
            sorted_tuples.append(t)
        sorted_tuples.sort()

        for i, (sbj,obj, pred) in enumerate(sorted_tuples):
            if self.verbose and i % self._print_every == 0:
                print('Processed %d tuples, found %d valid tuples.' % (i, len(src)))

            sbjent = self.parse_entity(sbj)
            rel = self.parse_relation(pred)
            objent = self.parse_entity(obj)
            processed = self.process_tuple((sbj, pred, obj), sbjent, rel, objent)
            if processed is None:
                # ignored
                continue
            # meta graph
            sbjclsid = _get_id(ent_classes, sbjent.cls)
            objclsid = _get_id(ent_classes, objent.cls)
            relclsid = _get_id(rel_classes, rel.cls)
            mg.add_edge(sbjent.cls, objent.cls, key=rel.cls)
            if self._insert_reverse:
                mg.add_edge(objent.cls, sbjent.cls, key='rev-%s' % rel.cls)
            # instance graph
            src_id = _get_id(entities, str(sbjent))
            if len(entities) > len(ntid):  # found new entity
                ntid.append(sbjclsid)
            dst_id = _get_id(entities, str(objent))
            if len(entities) > len(ntid):  # found new entity
                ntid.append(objclsid)
            src.append(src_id)
            dst.append(dst_id)
            etid.append(relclsid)

        src = np.asarray(src)
        dst = np.asarray(dst)
        ntid = np.asarray(ntid)
        etid = np.asarray(etid)
        ntypes = list(ent_classes.keys())
        etypes = list(rel_classes.keys())

        # add reverse edge with reverse relation
        if self._insert_reverse:
            if self.verbose:
                print('Adding reverse edges ...')
            newsrc = np.hstack([src, dst])
            newdst = np.hstack([dst, src])
            src = newsrc
            dst = newdst
            etid = np.hstack([etid, etid + len(etypes)])
            etypes.extend(['rev-%s' % t for t in etypes])

        hg = self.build_graph(mg, src, dst, ntid, etid, ntypes, etypes)
        idmap = F.asnumpy(hg.nodes["/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"].data[dgl.NID])
        glb2lcl = {glbid: lclid for lclid, glbid in enumerate(idmap)}

        def findidfn(ent):
            if ent not in entities:
                return None
            else:
                return glb2lcl[entities[ent]]

        self._hg = hg
        train_idx, test_idx, labels, num_classes = self.load_data_split(findidfn)

        train_mask = idx2mask(train_idx, self._hg.number_of_nodes("/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"))
        test_mask = idx2mask(test_idx, self._hg.number_of_nodes("/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"))
        labels = F.tensor(labels, F.data_type_dict['int64'])

        train_mask = generate_mask_tensor(train_mask)
        test_mask = generate_mask_tensor(test_mask)
        self._hg.nodes["/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"].data['train_mask'] = train_mask
        self._hg.nodes["/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"].data['test_mask'] = test_mask
        self._hg.nodes["/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"].data['labels'] = labels
        self._num_classes = num_classes

        # save for compatability
        self._train_idx = F.tensor(train_idx)
        self._test_idx = F.tensor(test_idx)
        self._labels = labels


    def build_graph(self, mg, src, dst, ntid, etid, ntypes, etypes):
        """Build the graphs

        Parameters
        ----------
        mg: MultiDiGraph
            Input graph
        src: Numpy array
            Source nodes
        dst: Numpy array
            Destination nodes
        ntid: Numpy array
            Node types for each node
        etid: Numpy array
            Edge types for each edge
        ntypes: list
            Node types
        etypes: list
            Edge types

        Returns
        -------
        g: DGLGraph
        """
        # create homo graph
        if self.verbose:
            print('Creating one whole graph ...')
        g = dgl.graph((src, dst))
        g.ndata[dgl.NTYPE] = F.tensor(ntid)
        g.edata[dgl.ETYPE] = F.tensor(etid)
        if self.verbose:
            print('Total #nodes:', g.number_of_nodes())
            print('Total #edges:', g.number_of_edges())

        # rename names such as 'type' so that they an be used as keys
        # to nn.ModuleDict
        etypes = [RENAME_DICT.get(ty, ty) for ty in etypes]
        mg_edges = mg.edges(keys=True)
        mg = nx.MultiDiGraph()
        for sty, dty, ety in mg_edges:
            mg.add_edge(sty, dty, key=RENAME_DICT.get(ety, ety))

        # convert to heterograph
        if self.verbose:
            print('Convert to heterograph ...')
        hg = dgl.to_heterogeneous(g,
                                  ntypes,
                                  etypes,
                                  metagraph=mg)
        if self.verbose:
            print('#Node types:', len(hg.ntypes))
            print('#Canonical edge types:', len(hg.etypes))
            print('#Unique edge type names:', len(set(hg.etypes)))
        return hg

    def load_data_split(self, ent2id):
        """Load data split

        Parameters
        ----------
        ent2id: func
            A function mapping entity to id
        root_path: str
            Root path containing the data

        Return
        ------
        train_idx: Numpy array
            Training set
        test_idx: Numpy array
            Testing set
        labels: Numpy array
            Labels
        num_classes: int
            Number of classes
        """
        label_dict = {}
        labels = np.zeros((self._hg.number_of_nodes("/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"),)) - 1
        train_idx = self.parse_idx_file(
            os.path.join('../data/deneme/trainingSet.tsv'),
            ent2id, label_dict, labels)
        test_idx = self.parse_idx_file(
            os.path.join('../data/deneme/testSet.tsv'),
            ent2id, label_dict, labels)
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        labels = np.array(labels)
        num_classes = len(label_dict)
        return train_idx, test_idx, labels, num_classes

    def parse_idx_file(self, filename, ent2id, label_dict, labels):
        """Parse idx files

        Parameters
        ----------
        filename: str
            File to parse
        ent2id: func
            A function mapping entity to id
        label_dict: dict
            Map label to label id
        labels: dict
            Map entity id to label id

        Return
        ------
        idx: list
            Entity idss
        """
        idx = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # first line is the header
                sample, label = self.process_idx_file_line(line)
                # person, _, label = line.strip().split('\t')
                ent = self.parse_entity(sample)
                entid = ent2id(str(ent))
                if entid is None:
                    print('Warning: entity "%s" does not have any valid links associated. Ignored.' % str(ent))
                else:
                    idx.append(entid)
                    lblid = _get_id(label_dict, label)
                    labels[entid] = lblid
        return idx

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join("../data/deneme/",
                                  self.save_name + '.bin')
        info_path = os.path.join("../data/deneme/",
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self._hg)


    def load(self):
        """load the graph list and the labels from disk"""
        graph_path = os.path.join("../data/deneme/",
                                  self.save_name + '.bin')
        info_path = os.path.join("../data/deneme/",
                                 self.save_name + '.pkl')
        graphs, _ = load_graphs(str(graph_path))

    def __getitem__(self, idx):
        r"""Gets the graph object
        """
        g = self._hg
        return g

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def graph(self):
        deprecate_property('dataset.graph', 'hg = dataset[0]')
        return self._hg

    @property
    def predict_category(self):
        return self._predict_category

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def train_idx(self):
        deprecate_property('dataset.train_idx', 'train_mask = g.ndata[\'train_mask\']')
        return self._train_idx

    @property
    def test_idx(self):
        deprecate_property('dataset.test_idx', 'train_mask = g.ndata[\'test_mask\']')
        return self._test_idx

    @property
    def labels(self):
        deprecate_property('dataset.labels', 'train_mask = g.ndata[\'labels\']')
        return self._labels

    @abc.abstractmethod
    def parse_entity(self, term):
        """Parse one entity from an RDF term.
        Return None if the term does not represent a valid entity and the
        whole tuple should be ignored.
        Parameters
        ----------
        term : rdflib.term.Identifier
            RDF term
        Returns
        -------
        Entity or None
            An entity.
        """
        pass

    @abc.abstractmethod
    def parse_relation(self, term):
        """Parse one relation from an RDF term.
        Return None if the term does not represent a valid relation and the
        whole tuple should be ignored.
        Parameters
        ----------
        term : rdflib.term.Identifier
            RDF term
        Returns
        -------
        Relation or None
            A relation
        """
        pass

    @abc.abstractmethod
    def process_tuple(self, raw_tuple, sbj, rel, obj):
        """Process the tuple.
        Return (Entity, Relation, Entity) tuple for as the final tuple.
        Return None if the tuple should be ignored.

        Parameters
        ----------
        raw_tuple : tuple of rdflib.term.Identifier
            (subject, predicate, object) tuple
        sbj : Entity
            Subject entity
        rel : Relation
            Relation
        obj : Entity
            Object entity
        Returns
        -------
        (Entity, Relation, Entity)
            The final tuple or None if should be ignored
        """
        pass

    @abc.abstractmethod
    def process_idx_file_line(self, line):
        """Process one line of ``trainingSet.tsv`` or ``testSet.tsv``.
        Parameters
        ----------
        line : str
            One line of the file
        Returns
        -------
        (str, str)
            One sample and its label
        """
        pass




    def __getitem__(self, idx):
        r"""Gets the graph object
        """
        g = self._hg
        return g

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def graph(self):
        deprecate_property('dataset.graph', 'hg = dataset[0]')
        return self._hg


class TwitterDataset(RDFGraphDataset):

    entity_prefix1 = 'http://www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests#'
    entity_prefix2 = 'http://dbpedia.org/resource/'
    relation_prefix = 'http://dbpedia.org/ontology/'



    def __init__(self, name=None, url=None,
                 raw_dir='../data/deneme/',
                 force_reload=False,
                 verbose=True):

        name = 'model_junior.rdf'

        super(TwitterDataset, self).__init__(name, url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def __getitem__(self, idx):
            r"""Gets the graph object

            Parameters
            -----------
            idx: int
                Item index, AIFBDataset has only one graph object

            Return
            -------
            :class:`dgl.DGLGraph`

                The graph contains:

                - ``ndata['train_mask']``: mask for training node set
                - ``ndata['test_mask']``: mask for testing node set
                - ``ndata['labels']``: mask for labels
            """
            return super(TwitterDataset, self).__getitem__(idx)


    def __len__(self):
        r"""The number of graphs in the dataset.

        Return
        -------
        int
        """
        return super(TwitterDataset, self).__len__()

    def parse_entity(self, term):
        if isinstance(term, rdf.Literal):
            return Entity(e_id=str(term), cls="_Literal")
        if isinstance(term, rdf.BNode):
            return None
        entstr = str(term)
        if entstr.startswith(self.entity_prefix1):
            sp = entstr.find("#")+1
            cl = entstr.find("/")
            print(Entity(e_id=entstr[sp:], cls=entstr[cl+1:sp-1]))
            return Entity(e_id=entstr[sp:], cls=entstr[cl+1:sp-1])
        elif entstr.startswith(self.entity_prefix2):
            cl = entstr.rindex("/")
            print(Entity(e_id=entstr[cl+1:], cls=entstr[:cl]))
            return Entity(e_id=entstr[cl+1:], cls=entstr[:cl])
        else:
            return None

    def parse_relation(self, term):
        if isinstance(term, rdf.Literal):
            #print(term)
            return Relation(cls="_Literal")
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            return Relation(cls=relstr.split('/')[4])
        else:
            relstr = relstr.split('/')[-1]
            return Relation(cls=relstr)

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        if sbj is None or rel is None or obj is None:
            return None
        return (sbj, rel, obj)

    def process_idx_file_line(self, line):
        _, person, _, label = line.strip().split('\t')
        return person, label



def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id


g=TwitterDataset()
g = g[0]

num_classes = TwitterDataset().num_classes
train_mask = g.nodes["/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"].data.pop('train_mask')
test_mask = g.nodes["/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"].data.pop('test_mask')
labels = g.nodes["/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests"].data.pop('labels').tolist()
import torch
train_idx = torch.nonzero(train_mask).squeeze().tolist()
test_idx = torch.nonzero(test_mask).squeeze().tolist()
edge_list = []
mat_dict = {}
for srctype, etype, dsttype in g.canonical_etypes:
        canonical_etypes = (srctype, etype, dsttype)
        edge_type = srctype.strip('_')+'||'+etype.strip('_')+'||' + dsttype.strip('_')
        mat_dict[edge_type] = g.adj(scipy_fmt='coo', etype=canonical_etypes)
        edge_list.append(edge_type)

info_dict = {'num_classes': num_classes, 'predict_category': "/www.semanticweb.org/gokce/ontologies/2021/11/twitter-interests", 'train_idx': train_idx,
                 'test_idx': test_idx, 'labels': labels, 'ntypes': g.ntypes, 'etypes': g.etypes, 'edge_list': edge_list}
import scipy.io
import scipy.sparse as sp
scipy.io.savemat('interests.mat', mat_dict)
import json
with open('interests_info.json', 'w') as f:
    json.dump(info_dict, f)