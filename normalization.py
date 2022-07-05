# Funções de normalização/denormalização dos dados
def Normalize(data):
    data_max = data.max()
    data_min = data.min()
    return (data-data_min)/(data_max-data_min)

def Denormalize(data_normalized, data):
    data_max = data.max()
    data_min = data.min()
    return data_normalized*(data_max-data_min) + data_min

