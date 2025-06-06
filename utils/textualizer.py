from utils.multiplex import Multiplex

def textualize_edges(mp: Multiplex, layer_name: bool = True, header: bool = True) -> list:
  text = ''

  for layer in mp.layers:
    if layer_name:
      text += layer['layer_name'] + '\n'

    if header:
      text += 'src\tweight\ttgt\n'

    for e in layer['graph'].edges(data=True):
      text += f"{e[0]}\t{e[2]['weight']}\t{e[1]}\n"
  
  return text

