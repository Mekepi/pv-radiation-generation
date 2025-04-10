from multiprocessing import Pool
from psutil import cpu_count
from time import perf_counter
from os import makedirs
from os.path import dirname, abspath
from pathlib import Path
from collections import defaultdict

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def write_cities(geocode:str, lines:list[str], header:str, pgr_path:Path) -> None:
    state:str = states[geocode[:2]]
    
    makedirs("%s\\gd-cities\\%s"%(pgr_path, state), exist_ok = True)

    with open("%s\\gd-cities\\%s\\[%s]empreendimentos-gd.csv"%(pgr_path, state, geocode), 'w', 10_485_760, encoding='ansi') as fout:
        fout.write(header)
        fout.writelines(lines)

def filter_per_city(file_path:Path, geocode_column:int) -> None:
    t0:float = perf_counter()
    pgr_path:Path = Path(dirname(abspath(__file__)))
    
    cities:defaultdict[str, list[str]] = defaultdict(list[str])

    with open("%s\\%s"%(dirname(abspath(__file__)), file_path), 'r', encoding='ansi') as fin:
        header:str = fin.readline()
        #print(*[(i, header.split('";"')[i]) for i in range(len(header.split('";"')))])
        
        for line in fin:
            geocode:str = line.split('";"')[geocode_column]
            cities[geocode].append(line)

    with Pool(cpu_count()) as p:
        p.starmap(write_cities, [[key, cities[key], header, pgr_path] for key in cities.keys()])
    
    print('execution time: %.2f'%(perf_counter()-t0))

if __name__ == '__main__':
    filter_per_city(Path("empreendimento-gd-unified-fixed-coords.csv"), 1)

# (0, '"CodEmpreendimento') (1, 'CodMunicipioIbge') (2, 'NumCoordNEmpreendimento') (3, 'NumCoordEEmpreendimento')
# (4, 'MdaPotenciaInstaladaKW') (5, 'MdaAreaArranjo') (6, 'QtdModulos') (7, 'MdaPotenciaModulos') (8, 'NomModeloModulo')
# (9, 'NomFabricanteModulo') (10, 'MdaPotenciaInversores') (11, 'NomModeloInversor"') (12, 'NomFabricanteInversor')
# (13, 'QtdUCRecebeCredito') (14, 'CodClasseConsumo') (15, 'DscSubGrupoTarifario') (16, 'SigModalidadeEmpreendimento')
# (17, 'SigTipoConsumidor') (18, 'NumCPFCNPJ') (19, 'CodCEP') (20, 'NumCNPJDistribuidora') (21, 'NomAgente')
# (22, 'NomSubEstacao') (23, 'NumCoordNSub') (24, 'NumCoordESub') (25, 'SigTipoGeracao') (26, 'DscPorte') (27, 'DthAtualizaCadastralEmpreend"\n')