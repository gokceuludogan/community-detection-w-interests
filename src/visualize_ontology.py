'''
!pip install pydotplus
!pip install graphviz
'''
import io
import pydotplus
import rdflib
from IPython.display import display, Image
from rdflib.tools.rdf2dot import rdf2dot

g = rdflib.Graph()
result = g.parse('interests.owl', format='application/rdf+xml')


def visualize(g):
    stream = io.StringIO()
    rdf2dot(g, stream, opts = {display})
    dg = pydotplus.graph_from_dot_data(stream.getvalue())
    png = dg.create_png()
    display(Image(png))

visualize(g)