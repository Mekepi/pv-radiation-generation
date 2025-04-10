import numpy as np
from time import perf_counter
from datetime import datetime
from os import listdir
from os.path import dirname, abspath
from pathlib import Path
from scipy.spatial import cKDTree
from psutil import cpu_count
from multiprocessing import Pool
from gzip import open as gsopen
from os import makedirs

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def curve_gen(data:list[str], orig_coord:list[float], coord:np.ndarray, loss:float) -> None:

    geocode:str = data[1]
    state:str = states[geocode[:2]]
    ceg:str = data[0]
    #analisar o que deve ser consiferado como power e determinar loss
    power:float = float(float('.'.join(data[4].split(','))))*1000*(1-loss)
    area:float = float('.'.join(data[6].split(',')))*1.65

    coords_path:Path = Path("%s\\data v5.3\\%s\\[%s]\\%s"%(dirname(abspath(__file__)), state, geocode,
                                                                next(f for f in listdir("%s\\data v5.3\\%s\\[%s]"%(dirname(abspath(__file__)), state, geocode)) if f[19:].startswith("(%.6f,%.6f)"%(coord[1], coord[0])))))
    
    with gsopen(coords_path, 'rt') as f:
        lines:list[str] = f.readlines()[10:-12]

    #AAAA-MM-DD" -> AAAAMMDD
    month:str = ''.join(data[27].split('-')[:-1])

    if (not(int(lines[0][:4])<=int(month[:4])<=int(lines[-1][:4]))):
        month = lines[-1][:4]+month[4:]

    if (state == "AC"):
        time_correction:int = 5
    elif (state == "AM"):
        time_correction = 4
    elif (geocode == "2605459"):
        time_correction = 2
    else:
        time_correction = 3

    lines = [line for line in lines if line.startswith(month[:6])]
    lines = lines[time_correction:]+lines[:time_correction]

    month_radiation:list[list[float]] = [[0,0] for _ in lines]
    for i in range(len(lines)):
        #time,Gb(i),Gd(i),Gr(i),H_sun,T2m,WS10m,Int
        elements:list[float] = [float(e) for e in lines[i].split(',')[1:]]
        month_radiation[i][0] = sum(elements[:3])
        tc:float = (elements[4]+(sum(elements[:3])*25/800)-25)
        tc = tc if tc>0 else 0
        month_radiation[i][1] = 1-0.95*0.97*0.98*(1-0.0045*tc)*(1-0.005*(datetime.now().year-int(month[:4])))
    
    #G,H_sun,T2m,WS10m,Int
    month_generation:list[list[float]] = [[month_radiation[i][0]*month_radiation[i][1]*power/1000, month_radiation[i][1]] for i in range(len(month_radiation))]
    

    #hourly_radiation = [float(lines[i].split(',')[2]) for i in range(len(lines)) if (lines[i].startswith(day+data[0][9:11]))]
    
    #month_radiation:list[float] = [sum([float(lines[i].split(',')[j]) for j in [1,2,3]]) for i in range(len(lines)) if (lines[i].startswith(day))]

    hourly_radiation:list[float] = [sum([month_radiation[k][0] for k in range(i, len(month_radiation), 24)])/(len(month_radiation)//24) for i in range(24)]
    hourly_generation:list[float] = [sum([month_generation[k][0] for k in range(i, len(month_generation), 24)])/(len(month_generation)//24) for i in range(24)]

    # v1:
    """ radiation_energy:float = sum(hourly_radiation)*1.0 # [Wh]
    correction_factor:float = radiation_energy/power # potência de instalação pico com loss [w] / potência pico de radição*area [w/m²*m² == w]
    generated_energy:float = radiation_energy*correction_factor # [kWh]
    produced_hourly:list[float] = [hourly_radiation[i]*correction_factor for i in range(len(hourly_radiation))] """

    # v2:
    radiation_energy:float = sum(hourly_radiation)/1000 # [kWh/m²]
    generated_energy:float = sum(hourly_generation)/1000 # [kWh]
    loss = sum([e[1] for e in month_generation])/len(month_generation)
    correction_factor1:float = generated_energy/radiation_energy
    correction_factor2:float = power*(1-loss)/(radiation_energy*1000)


    import matplotlib.pyplot as plt
    from matplotlib import use
    use("Agg")

    makedirs("%s\\curves\\%s\\[%s]"%(dirname(abspath(__file__)), state, geocode), exist_ok=True)

    plt.plot(range(24), [hourly_radiation[i]/1000 for i in range(24)], label="Radiation Energy Curve", color="black")
    #plt.bar(range(24), [hourly_generation[i]/1000 for i in range(24)], label="Produced Energy 1", color="#DB5C1F")
    #plt.bar(range(24), [hourly_radiation[i]*correction_factor2/1000 for i in range(24)], label="Produced Energy 2", color="blue")
    plt.title(ceg)
    plt.suptitle("Power curve at (%f,%f), [%s]\n\nArea: %.0f m²    Power: %.0f kW    Loss: %.2f %%"%(orig_coord[1], orig_coord[0], geocode, area, power/1000, loss*100))
    plt.xlabel("Time [Hour]\n\nFactors: (%.3f, %.3f)    Radiation Energy: %.2f kWh/m²    Produced Energy: %.2f kWh"%(correction_factor1, correction_factor2, radiation_energy, generated_energy))
    plt.ylabel("Power [kW]")
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s\\curves\\%s\\[%s]\\%s.png"%(dirname(abspath(__file__)), state, geocode, ceg), backend='Agg', dpi=200)

    plt.close()

def gds_generation_curve(sts:list[str] = [], geocodes:list[str] = [], loss:float=0.0) -> None:
    t0:float = perf_counter()

    brasil_coords:str = "%s\\%s"%(dirname(abspath(__file__)),
                                   next(f for f in listdir(dirname(abspath(__file__))) if f.startswith("Brasil")))

    with Pool(cpu_count()*2) as p:

        for state in listdir("%s\\gd-cities"%(dirname(abspath(__file__)))):

            if ((sts and not(state[:2] in sts)) or (geocodes and not(state[:2] in [states[s[:2]] for s in geocodes]))):
                continue

            state_coords:str = "%s\\%s"%(brasil_coords, next(f for f in listdir(brasil_coords) if f.startswith(state)))

            for city in listdir("%s\\gd-cities\\%s"%(dirname(abspath(__file__)), state)):

                if (geocodes and not(city[1:8] in geocodes)):
                    continue

                with open("%s\\gd-cities\\%s\\%s"%(dirname(abspath(__file__)), state, city), 'r', encoding='ansi') as file:
                    file.readline()
                    lines:list[str] = file.readlines()

                city_coords_path:str = "%s\\%s"%(state_coords, next(f for f in listdir(state_coords) if f.startswith(city[:9])))
                city_coords:np.ndarray = np.array(np.loadtxt(city_coords_path, delimiter=',', ndmin=2))

                failty_coord:list[str] = []
                gd_coords:list[tuple[float, float]] = []
                for line in lines:
                    if (line.split('";"')[3] != ',' and line.split('";"')[2] != ','):
                        gd_coords.append((float('.'.join(line.split('";"')[3].split(','))), float('.'.join(line.split('";"')[2].split(',')))))
                        continue
                    failty_coord.append(line)

                if failty_coord:
                    with open("%s\\failty_coord.csv"%(dirname(abspath(__file__))), 'a', encoding='ansi') as f:
                        f.writelines(failty_coord)
                
                distances:list[float]
                idxs:list[int]
                distances, idxs = cKDTree(city_coords).query(gd_coords, 1, workers=-1) # type: ignore

                faridxs:list[str] = ['%s %s %.2f    %s'%(str(gd_coords[i]),str(city_coords[idxs[i]]),distances[i],lines[i]) for i in range(len(distances)) if distances[i]>=0.03]

                if faridxs:
                    makedirs("%s\\Too Far Coords\\%s"%(dirname(abspath(__file__)), state), exist_ok=True)
                    with open("%s\\Too Far Coords\\%s\\%s-too-far.csv"%(dirname(abspath(__file__)), state, city[:9]), 'w', 1024*1024*256, encoding='ansi') as f:
                        f.write("source coord;closest timeseries coord;distance;line")
                        f.writelines(faridxs)
                        

                """ cts = np.concatenate((gd_coords, city_coords[idxs], np.reshape(distances, [len(distances),1])), 1)
                print(cts, sep="\n") """
                
                p.starmap(curve_gen, [[line[1:-2].split('";"'), orig_coord, closest_timeseries, loss] for (line, orig_coord, closest_timeseries) in zip(lines, gd_coords, city_coords[idxs])])

    print(perf_counter()-t0)

if __name__ == "__main__":
    gds_generation_curve(geocodes=['3501608'])

# (0, '"CodEmpreendimento') (1, 'CodMunicipioIbge') (2, 'NumCoordNEmpreendimento') (3, 'NumCoordEEmpreendimento')
# (4, 'MdaPotenciaInstaladaKW') (5, 'MdaAreaArranjo') (6, 'QtdModulos') (7, 'MdaPotenciaModulos') (8, 'NomModeloModulo')
# (9, 'NomFabricanteModulo') (10, 'MdaPotenciaInversores') (11, 'NomModeloInversor"') (12, 'NomFabricanteInversor')
# (13, 'QtdUCRecebeCredito') (14, 'CodClasseConsumo') (15, 'DscSubGrupoTarifario') (16, 'SigModalidadeEmpreendimento')
# (17, 'SigTipoConsumidor') (18, 'NumCPFCNPJ') (19, 'CodCEP') (20, 'NumCNPJDistribuidora') (21, 'NomAgente')
# (22, 'NomSubEstacao') (23, 'NumCoordNSub') (24, 'NumCoordESub') (25, 'SigTipoGeracao') (26, 'DscPorte') (27, 'DthAtualizaCadastralEmpreend"\n')