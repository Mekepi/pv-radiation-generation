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
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D #type: ignore

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def avarage_month_radiation_generation(timeseries:list[str], state:str, geocode:str, ceg:str, orig_coord:list[float], data:list[str], time_correction:int, city_plot_folder:Path):
    loss:float = 0.14
    power:float = float(float('.'.join(data[4].split(','))))*1000*(1-loss)
    panels:int = int(data[6])
    area:float = float('.'.join(data[5].split(',')))
    
    #AAAA-MM-DD" -> AAAAMMDD
    month:str = ''.join(data[27].split('-')[:-1])
    if (not(int(timeseries[0][:4])<=int(month[:4])<=int(timeseries[-1][:4]))):
        month = timeseries[-1][:4]+month[4:]

    timeseries = [line for line in timeseries if line.startswith(month[:6])]
    timeseries = timeseries[time_correction:]+timeseries[:time_correction]

    month_radiation:list[list[float]] = [[0,0] for _ in timeseries]
    for i in range(len(timeseries)):
        #time,Gb(i),Gd(i),Gr(i),H_sun,T2m,WS10m,Int
        elements:list[float] = [float(e) for e in timeseries[i].split(',')[1:]]
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

    # v2:
    radiation_energy:float = sum(hourly_radiation)/1000*area # [kWh/m²]
    generated_energy1:float = sum(hourly_generation)/1000 # [kWh]
    loss1:float = sum([e[1] for e in month_generation])/len(month_generation)
    correction_factor1:float = power*loss1/1000 #generated_energy1/radiation_energy
    loss2:float = 0.14
    correction_factor2:float = power*(1-loss2)/(radiation_energy*1000)*area
    generated_energy2:float = sum(hourly_radiation)*correction_factor2/1000


    import matplotlib.pyplot as plt
    from matplotlib import use
    from matplotlib.axes import Axes
    use("Agg")

    ax1:Axes
    fig, ax1 = plt.subplots()

    ax2:Axes = ax1.twinx() #type: ignore
    
    ax2.plot(range(24), [hourly_radiation[i]/1000 for i in range(24)], label="Irradiance Curve", color="black") #radiation
    ax1.bar(range(24), [hourly_generation[i]/1000 for i in range(24)], label="Method 1", color="#24B351") # 1sr formula
    ax1.bar(range(24), [hourly_radiation[i]*correction_factor2/1000 for i in range(24)], label="Method 2", color="#1F2792", alpha=0.70) #2nd formula
    plt.title(ceg)
    plt.suptitle("PV Yield at (%f,%f), [%s]\n\nArea: %.0f m²    Panels: %i    Power: %.0f kW"%(orig_coord[1], orig_coord[0], geocode, area, panels, power/1000))
    ax1.set_xlabel("Time [Hour]\n\nRadiation Energy: %.2f kWh\n\nMethod 1 - Factor: %.3f    Produced Energy: %.2f kWh    Loss: %.2f %%\n\nMethod 2 - Factor: %.3f    Produced Energy: %.2f kWh    Loss: %.2f %%"%(radiation_energy, correction_factor1, generated_energy1, loss1*100, correction_factor2, generated_energy2, loss2*100))
    ax1.set_ylabel("Energy [kWh]", color='#24B351')
    ax2.set_ylabel("Irradiance [kWh/m²]", color='black')
    
    if ax1.set_ylim()[1] > ax2.set_ylim()[1]:
        ax2.set_ylim(ax1.set_ylim())
    else:
        ax1.set_ylim(ax2.set_ylim())
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.savefig("%s\\%s.png"%(city_plot_folder, ceg), backend='Agg', dpi=200)

    plt.close()


def average_year_radiation(timeseries:list[str], orig_coord:list[float], geocode:str, ceg:str, time_correction:int, city_plot_folder:Path):
    year:defaultdict[str, list[float]] = defaultdict(list[float])
    for line in timeseries:
        year[line.split(',')[0][4:8]].append(sum([float(j) for j in line.split(',')[1:4]]))

    x:np.ndarray = np.array(list(range(1, 367))).T
    y:np.ndarray = np.array(list(range(24)))
    #z:np.ndarray = np.array()
    z = np.asarray([[sum([year[day][j] for j in range(i, len(year[day]), 24)])/(len(year[day])/24) for i in range(24)] for day in sorted(year.keys())])
    z = np.concatenate([z[:, time_correction:], z[:, :time_correction]], axis=1)
    #print(year['0101'])
    #print(x.shape, y.shape, z.shape)
    
    import matplotlib.pyplot as plt
    from matplotlib import use
    use("Agg")

    ax:Axes3D = plt.axes(projection='3d')
    X, Y = np.meshgrid(x,y)
    
    ax.plot_surface(X.T, Y.T, z, cmap='viridis')
    ax.view_init(20, -50, 0)
    ax.set_xlabel('Day of the Year [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Solar Irradiance [kW/m²]')
    ax.set_title("Hourly Solar Radiation Across the Year\n\n%s\n(%f,%f) [%s]"%(ceg, orig_coord[1], orig_coord[0], geocode))
    plt.tight_layout()
    plt.savefig("%s\\%s-3D-year-radiation.png"%(city_plot_folder, ceg), backend='Agg', dpi=200)
    plt.close()
    #22, -33, 0

def curve_gen(data:list[str], orig_coord:list[float], coord:np.ndarray, loss:float) -> None:

    timeseries_coords_folder:Path = Path('%s\\data\\timeseries'%(Path(dirname(abspath(__file__))).parent))

    geocode:str = data[1]
    state:str = states[geocode[:2]]
    ceg:str = data[0]

    if (state == "AC"):
        time_correction:int = 5
    elif (state == "AM"):
        time_correction = 4
    elif (geocode == "2605459"):
        time_correction = 2
    else:
        time_correction = 3

    coords_path:Path = Path("%s\\%s\\[%s]\\%s"%(timeseries_coords_folder, state, geocode,
                                                                next(f for f in listdir("%s\\%s\\[%s]"%(timeseries_coords_folder, state, geocode)) if f[19:].startswith("(%.6f,%.6f)"%(coord[1], coord[0])))))
    
    with gsopen(coords_path, 'rt') as f:
        lines:list[str] = f.readlines()[9:-12]
    
    city_plot_folder:Path = Path("%s\\outputs\\plot\\%s\\[%s]"%(Path(dirname(abspath(__file__))).parent, state, geocode))
    makedirs("%s"%(city_plot_folder), exist_ok=True)
    
    average_year_radiation(lines, orig_coord, geocode, ceg, time_correction, city_plot_folder)
    avarage_month_radiation_generation(lines, state, geocode, ceg, orig_coord, data, time_correction, city_plot_folder)

def gds_generation_curve(sts:list[str] = [], geocodes:list[str] = [], loss:float=0.0) -> None:
    t0:float = perf_counter()

    data_folder:Path = Path(dirname(abspath(__file__))).parent

    ventures_folder:Path = Path('%s\\data\\ventures'%(data_folder))

    timeseries_coords_folder:Path = Path('%s\\data\\timeseries_coords'%(data_folder))

    with Pool(cpu_count()*2) as p:

        for state in listdir(ventures_folder):

            if ((sts and not(state[:2] in sts)) or (geocodes and not(state[:2] in [states[s[:2]] for s in geocodes]))):
                continue

            state_timeseries_coords_folder:str = "%s\\%s"%(timeseries_coords_folder, next(f for f in listdir(timeseries_coords_folder) if f.startswith(state)))

            for city in listdir('%s\\%s'%(ventures_folder, state)):

                if (geocodes and not(city[1:8] in geocodes)):
                    continue

                with open("%s\\%s\\%s"%(ventures_folder, state, city), 'r', encoding='utf-8') as file:
                    file.readline()
                    lines:list[str] = file.readlines()

                city_timeseries_coords_file:Path = Path("%s\\%s"%(state_timeseries_coords_folder, next(f for f in listdir(state_timeseries_coords_folder) if f.startswith(city[:9]))))
                city_coords:np.ndarray = np.array(np.loadtxt(city_timeseries_coords_file, delimiter=',', ndmin=2))

                failty_coord:list[str] = []
                city_ventures_coords:list[tuple[float, float]] = []
                for line in lines:
                    if (line.split('";"')[3] != ',' and line.split('";"')[2] != ','):
                        city_ventures_coords.append((float('.'.join(line.split('";"')[3].split(','))), float('.'.join(line.split('";"')[2].split(',')))))
                        continue
                    failty_coord.append(line)

                if failty_coord:
                    with open("%s\\failty_coord.csv"%(data_folder), 'a', encoding='utf-8') as f:
                        f.writelines(failty_coord)
                
                distances:list[float]
                idxs:list[int]
                distances, idxs = cKDTree(city_coords).query(city_ventures_coords, 1, workers=-1) # type: ignore

                faridxs:list[str] = ["(%7.2f,%6.2f) (%11.6f,%6.6f) %6.2f    %s"%(*city_ventures_coords[i],*city_coords[idxs[i]],distances[i],lines[i]) for i in range(len(distances)) if distances[i]>=0.03]
                
                if faridxs:
                    makedirs("%s\\outputs\\Too Far Coords\\%s"%(data_folder, state), exist_ok=True)
                    with open("%s\\outputs\\Too Far Coords\\%s\\%s-too-far.csv"%(data_folder, state, city[:9]), 'w', 1024*1024*256, encoding='utf-8') as f:
                        f.write("source coord;closest timeseries coord;distance;line\n")
                        f.writelines(faridxs)
                        

                """ cts = np.concatenate((city_ventures_coords, city_coords[idxs], np.reshape(distances, [len(distances),1])), 1)
                print(cts, sep="\n") """
                
                p.starmap(curve_gen, [[line[1:-2].split('";"'), orig_coord, closest_timeseries, loss] for (line, orig_coord, closest_timeseries) in zip(lines, city_ventures_coords, city_coords[idxs])])

    print(perf_counter()-t0)

if __name__ == "__main__":
    gds_generation_curve(geocodes=['3550308'])

# (0, '"CodEmpreendimento') (1, 'CodMunicipioIbge') (2, 'NumCoordNEmpreendimento') (3, 'NumCoordEEmpreendimento')
# (4, 'MdaPotenciaInstaladaKW') (5, 'MdaAreaArranjo') (6, 'QtdModulos') (7, 'MdaPotenciaModulos') (8, 'NomModeloModulo')
# (9, 'NomFabricanteModulo') (10, 'MdaPotenciaInversores') (11, 'NomModeloInversor"') (12, 'NomFabricanteInversor')
# (13, 'QtdUCRecebeCredito') (14, 'CodClasseConsumo') (15, 'DscSubGrupoTarifario') (16, 'SigModalidadeEmpreendimento')
# (17, 'SigTipoConsumidor') (18, 'NumCPFCNPJ') (19, 'CodCEP') (20, 'NumCNPJDistribuidora') (21, 'NomAgente')
# (22, 'NomSubEstacao') (23, 'NumCoordNSub') (24, 'NumCoordESub') (25, 'SigTipoGeracao') (26, 'DscPorte') (27, 'DthAtualizaCadastralEmpreend"\n')