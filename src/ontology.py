import logging
import rdflib
from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF , XSD, OWL


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

ONTOLOGY_NAME = 'twitter-interests'

class Ontology:
  def __init__(self, filename):
    self.graph = Graph()
    self.graph.parse(filename, format='application/rdf+xml')
    self.graph.bind("owl", OWL)
    self.graph.bind("foaf", FOAF)

    ontology_uriref = [s for s, p, o in self.graph.triples((None, RDF.type, OWL.Ontology))][0]
    ontology_uri_str = str(ontology_uriref)

    self.ns_interests = rdflib.Namespace(ontology_uri_str)
    self.ns_dbo = rdflib.Namespace("http://dbpedia.org/ontology/")

    self.graph.bind('dbo', self.ns_dbo) 
    self.graph.bind(ONTOLOGY_NAME, self.ns_interests)

    self.nm = self.graph.namespace_manager
    self.nm.bind(ONTOLOGY_NAME, self.ns_interests)
  
    self._map_classes()
    self._map_properties()
  
  def _map_classes(self):
    classes = [s for s, p, o in self.graph.triples((None, RDF.type , OWL.Class))]
    self.class_mapping = {str(class_).split('/')[-1].split('#')[-1]: class_ for class_ in classes}
    
  def _map_properties(self):
    properties = [s for s, p, o in self.graph.triples((None, RDF.type  , OWL.ObjectProperty))]
    self.property_mapping = {str(p).split('/')[-1].split('#')[-1]: p for p in properties}

  def add_node(self, node, n_type):
    if (node, RDF.type, n_type) in self.graph:
      logging.info(f'{str(node)} already exists!')
    else:
      self.graph.add((node, RDF.type, n_type))

  def add_interaction(self, user1, user2, weight, interaction_type):
    user1_uri = self.ns_interests[user1] 
    user2_uri = self.ns_interests[user2] 

    self.add_node(user1_uri, self.ns_interests.User)
    self.add_node(user2_uri, self.ns_interests.User)
    
    interaction_level_id = f'{user1}-{user2}-{interaction_type}' 
    il_uri = self.ns_interests[interaction_level_id]

    interacts = self.property_mapping['interacts']
    interaction_scale = self.property_mapping['interactionScale']
    interaction = self.property_mapping[interaction_type]

    self.graph.add((user1_uri, interacts, il_uri))
    self.graph.add((il_uri, interaction, user2_uri))
    self.graph.add((il_uri, interaction_scale, Literal(weight)))

  def add_interaction_list(self, triples, interaction_type):
    for user1, user2, weight in triples:
        self.add_interaction(user1, user2, weight, interaction_type)

  def add_entity_list(self, triples):
    for user, interest_type, interest in triples:
        self.add_interest(user, interest_type, interest)
  
  
  def add_interest(self, user, interest_type, interest):
    user_uri = self.ns_interests[user] 
    self.add_node(user_uri, self.ns_interests.User)

    interest_uri = URIRef(interest) 

    self.add_node(interest_uri, self.class_mapping[interest_type])

    has_interest = self.property_mapping['hasInterest']
    self.graph.add((user_uri, has_interest, interest_uri))

  def save(self, output_file):
    self.graph.serialize(destination=output_file, format='application/rdf+xml')

  def query(self, q):
    return self.graph.query(q)

  def get_all_interactions(self, interaction_type='mention'):
    q = f'''
    prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns/> 
    prefix {ONTOLOGY_NAME}: <{str(self.ns_interests)}>
    SELECT DISTINCT ?user ?mentionedUser ?weight
    WHERE {{
      ?user {ONTOLOGY_NAME}:interacts  ?il .
      ?il {ONTOLOGY_NAME}:{interaction_type} ?mentionedUser .
      ?il {ONTOLOGY_NAME}:interactionScale ?weight.
    }}

    '''
    qres = self.query(q)
    return [(row.user, row.mentionedUser, row.weight) for row in qres]
  
  def get_user_interests(self, user):
    q = f'''
    prefix {ONTOLOGY_NAME}: <{str(self.ns_interests)}>
    SELECT DISTINCT ?topic
    WHERE {{
      {ONTOLOGY_NAME}:{user} {ONTOLOGY_NAME}:hasInterest  ?topic .
    }}

    '''
    qres = self.query(q)
    return [item.topic for item in qres]
  
  def get_class_based_interests(self):
    q = f'''
    prefix {ONTOLOGY_NAME}: <{str(self.ns_interests)}>
    SELECT DISTINCT ?user1 ?interest1 ?user2 ?interest2 ?interestType
    WHERE {{
      ?user1 {ONTOLOGY_NAME}:hasInterest ?interest1 .
      ?interest1 a ?interestType .
      ?user2 {ONTOLOGY_NAME}:hasInterest ?interest2 . 
      ?interest2 a ?interestType . 
      FILTER(?user1 != ?user2) .
    }}

    '''
    qres = self.query(q)
    return [(item.user1, item.interest1, item.user2, item.interest2, item.interestType) for item in qres]

  def get_all_interests(self):
    q = f'''
    prefix {ONTOLOGY_NAME}: <{str(self.ns_interests)}>
    SELECT DISTINCT ?user ?interest ?interestType
    WHERE {{
      ?user {ONTOLOGY_NAME}:hasInterest  ?interest .
      ?interest a ?interestType . 
    }}

    '''
    qres = self.query(q)
    return [(item.user, item.interest, item.interestType) for item in qres]
  
  def get_all_users(self):
    q = f'''
    prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns/> 
    prefix {ONTOLOGY_NAME}: <{str(self.ns_interests)}>
    SELECT DISTINCT ?user 
    WHERE {{
      ?user a {ONTOLOGY_NAME}:User .
    }}
    '''
    qres = self.query(q)
    return [item.user for item in qres]

# ontology = Ontology('interests-v3.owl')
# ontology.add_interaction('gokce', 'suzan', 1.0, 'mention')
# ontology.add_interaction('gokce', 'idil', 0.1, 'retweet')
# ontology.save('populated.owl')
# ontology.get_all_interactions('mention')
# ontology.get_all_interactions('retweet')
# ontology.get_all_users()
