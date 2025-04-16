import numpy as np
from time import perf_counter
from datetime import datetime
from os import listdir
from os.path import dirname, abspath
from pathlib import Path
from scipy.spatial import cKDTree
from psutil import cpu_count
from multiprocessing import Pool
from gzip import open as gzopen
from os import makedirs
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D #type: ignore

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def month_radiation_plot_old(timeseries:list[str], state:str, geocode:str, ceg:str, orig_coord:list[float], data:list[str], time_correction:int, city_plot_folder:Path):
    power:float = float(float('.'.join(data[4].split(','))))*1000
    panels:int = int(data[6])
    area:float = float('.'.join(data[5].split(',')))
    install_year:int = int(''.join(data[27][:4]))
    
    #AAAA-MM-DD" -> AAAAMMDD
    month:str = ''.join(data[27].split('-')[:-1])
    """ if (not(int(timeseries[0][:4])<=int(month[:4])<=int(timeseries[-1][:4]))):
        month = timeseries[-1][:4]+month[4:] """

    timeseries = [line for line in timeseries if line.startswith(month[:6])]
    timeseries = timeseries[time_correction:]+timeseries[:time_correction]

    month_radiation:list[list[float]] = [[0,0] for _ in timeseries]
    for i in range(len(timeseries)):
        #time,Gb(i),Gd(i),Gr(i),H_sun,T2m,WS10m,Int
        elements:list[float] = [float(e) for e in timeseries[i].split(',')[1:]]
        month_radiation[i][0] = sum(elements[:3])
        tc:float = (elements[4]+(sum(elements[:3])*25/800)-25)
        tc = tc if tc>0 else 0
        month_radiation[i][1] = 1-0.95*0.97*0.98*(1-0.0045*tc)*(1-0.005*(datetime.now().year-install_year))
    
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
    print('%.2f'%(1-loss1))
    correction_factor1:float = power*(1-loss1)/1000 #generated_energy1/radiation_energy
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

def loss(g:float, t:float, install_year:int) -> float:
    Tc:float = (t+g*25/800)
    Tc = Tc if Tc>25 else 25

    return 1-0.95*0.97*0.98*(1-0.005*(datetime.now().year-install_year))*(1 - 0.0045*(Tc-25))

def day_radiation_plot(day:np.ndarray, orig_coord:list[float], data:list[str], city_plot_folder:Path, ti:int):
    ceg:str = data[0]
    geocode:str = data[1]
    power:float = float(float('.'.join(data[4].split(','))))*1000
    panels:int = int(data[6])
    area:float = float('.'.join(data[5].split(',')))
    install_year:int = int(''.join(data[27][:4]))

    plot_type:str = ['Average Year Max Mounth', 'Average Year Min Mounth', 'Last Year Max Mounth', 'Last Year Min Mounth'][ti]
    
    #G,H_sun,T2m,WS10m,Int
    """ month_generation:list[list[float]] = [[month_radiation_temp[i][0]*loss(month_radiation_temp[i][0], month_radiation_temp[i][1], install_year)*power/1000, loss(month_radiation_temp[i][0], month_radiation_temp[i][1], install_year)] for i in range(len(month_radiation_temp))]

    hourly_radiation:list[float] = [sum([month_radiation_temp[k][0] for k in range(i, len(month_radiation_temp), 24)])/(len(month_radiation_temp)//24) for i in range(24)]
    hourly_generation:list[float] = [sum([month_generation[k][0] for k in range(i, len(month_generation), 24)])/(len(month_generation)//24) for i in range(24)] """

    hourly_radiation:np.ndarray = day[:, 0]
    hourly_generation:np.ndarray = np.asarray([[h[0]*(1-loss(h[0], h[1], install_year))*power/1000, loss(h[0], h[1], install_year)] for h in day])

    print(str([[(1-loss(h[0], h[1], install_year))*power/1000, loss(h[0], h[1], install_year)] for h in day]))

    # v2:
    radiation_energy:float = sum(hourly_radiation)/1000*area # [kWh]

    loss1:float = np.sum(hourly_generation[:, 1])/24
    correction_factor1:float = power*(1-loss1)/1000
    generated_energy1:float = sum(hourly_generation[:, 0])/1000 # [kWh]

    """ loss2:float = 0.14
    print([power*(1-loss2)/h for h in hourly_radiation if h>0])
    correction_factor2:float = sum([power*(1-loss2)/h for h in hourly_radiation if h>0])/hourly_radiation[hourly_radiation>0].shape[0]
    generated_energy2:float = sum(hourly_radiation)*correction_factor2/1000 """


    import matplotlib.pyplot as plt
    from matplotlib import use
    from matplotlib.axes import Axes
    use("Agg")

    ax1:Axes
    fig, ax1 = plt.subplots()

    ax2:Axes = ax1.twinx() #type: ignore
    
    ax2.plot(range(24), hourly_radiation, label="Irradiance Curve", color="black") #radiation
    ax1.bar(range(24), hourly_generation[:,0], label="Method 1", color="#24B351") # 1sr formula
    #ax1.bar(range(24), hourly_radiation*correction_factor2, label="Method 2", color="#1F2792", alpha=0.70) #2nd formula
    plt.title("%s\n%s"%(ceg, plot_type))
    plt.suptitle("PV Yield at (%f,%f), [%s]\n\nArea: %.0f m²    Panels: %i    Power: %.0f kW"%(orig_coord[1], orig_coord[0], geocode, area, panels, power/1000))
    ax1.set_xlabel("Time [Hour]\n\nRadiation Energy: %.2f kWh\n\nMethod 1 - Factor: %.3f    Produced Energy: %.2f kWh    Loss: %.2f %%"%(radiation_energy, correction_factor1, generated_energy1, loss1*100))
    ax1.set_ylabel("Energy [kWh]", color='#24B351')
    ax2.set_ylabel("Irradiance [kWh/m²]", color='black')
    
    if ax1.set_ylim()[1] > ax2.set_ylim()[1]:
        ax2.set_ylim(ax1.set_ylim())
    else:
        ax1.set_ylim(ax2.set_ylim())
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.savefig("%s\\%s-%s.png"%(city_plot_folder, ceg, plot_type), backend='Agg', dpi=200)

    plt.close()

def year_radiation_plot(Z:np.ndarray, orig_coord:list[float], geocode:str, ceg:str, city_plot_folder:Path, ti:int):
    plot_type:str = ["Average Year", "Last Year"][ti]

    import matplotlib.pyplot as plt
    from matplotlib import use
    use("Agg")

    x:np.ndarray = np.array(list(range(1, 367)))
    y:np.ndarray = np.array(list(range(24)))

    Y, X = np.meshgrid(y, x)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(20, -50, 0)
    ax.set_xlabel('Day of the Year [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Solar Irradiance [kW/m²]')
    ax.set_title("Hourly Solar Radiation Across the Year\n%s\n\n%s\n(%f,%f) [%s]"%(plot_type, ceg, orig_coord[1], orig_coord[0], geocode))
    plt.tight_layout()
    plt.savefig("%s\\%s-3D-year-radiation-%s.png"%(city_plot_folder, ceg, plot_type), backend='Agg', dpi=200)
    plt.close()

def curves_gen(data:list[str], orig_coord:list[float], coord:np.ndarray) -> None:

    timeseries_folder:Path = Path('%s\\data\\timeseries'%(Path(dirname(abspath(__file__))).parent))

    ceg:str = data[0]
    geocode:str = data[1]
    state:str = states[geocode[:2]]

    if (state == "AC"):
        time_correction:int = 5
    elif (state == "AM"):
        time_correction = 4
    elif (geocode == "2605459"):
        time_correction = 2
    else:
        time_correction = 3

    timeseries_path:Path = Path("%s\\%s\\[%s]\\%s"%(timeseries_folder, state, geocode,
                                                                next(f for f in listdir("%s\\%s\\[%s]"%(timeseries_folder, state, geocode)) if f[19:].startswith("(%.6f,%.6f)"%(coord[1], coord[0])))))
    
    with gzopen(timeseries_path, 'rt', encoding='utf-8') as f:
        lines:list[str] = f.readlines()[9:-12]

    last_year:list[list[list[float]]] = []
    last_year_value:str = lines[-1][:4]
    avarege_year:defaultdict[str, list[list[float]]] = defaultdict(list[list[float]])
    day:list[list[float]] = []
    for line in lines:
        spline:list[str] = line.split(',')
        avarege_year[spline[0][4:8]].append([sum([float(j) for j in spline[1:4]]), float(spline[5])])
        if (spline[0][:4] != last_year_value):
            continue
        day.append([sum([float(j) for j in spline[1:4]]), float(spline[5])])
        if (len(day) == 24):
            last_year.append(day)
            day = []

    #print(*[len(avarege_year[day][0]) for day in sorted(avarege_year.keys())])

    Z:np.ndarray = np.asarray([[[sum([avarege_year[day][j][0] for j in range(i, len(avarege_year[day]), 24)])/(len(avarege_year[day])/24),
                                 sum([avarege_year[day][j][1] for j in range(i, len(avarege_year[day]), 24)])/(len(avarege_year[day])/24)] for i in range(24)] for day in sorted(avarege_year.keys())])
    Z = np.concatenate([Z[:, time_correction:], Z[:, :time_correction]], axis=1)

    if (len(last_year) == 365):
        Zl:np.ndarray = np.asarray(last_year[:60]+last_year[60:61]+last_year[60:])
    else:
        Zl = np.asarray(last_year)
    Zl = np.concatenate([Zl[:, time_correction:], Zl[:, :time_correction]], axis=1)

    
    city_plot_folder:Path = Path("%s\\outputs\\plot\\%s\\[%s]"%(Path(dirname(abspath(__file__))).parent, state, geocode))
    makedirs("%s"%(city_plot_folder), exist_ok=True)
    
    year_radiation_plot(Z[:,:,0], orig_coord, geocode, ceg, city_plot_folder, 0)
    year_radiation_plot(Zl[:,:,0], orig_coord, geocode, ceg, city_plot_folder, 1)

    t1=perf_counter()
    # average day of the mounth with average max and min radiation in an avearage year 
    max_average_mounth:np.ndarray = np.zeros([1,24,2])
    min_average_mounth:np.ndarray = np.ones([1,24,2])*np.inf

    # average day of the mounth with average max and min radiation last year 
    max_last_year_mounth:np.ndarray = np.zeros([1,24,2])
    min_last_year_mounth:np.ndarray = np.ones([1,24,2])*np.inf

    days_of_mounths:list[int] = [31,29,31,30,31,30,31,31,30,31,30,31]
    i:int = 0
    for d in days_of_mounths:
        current_sum:float = np.sum(np.sum(Z[i:i+d, :, 0], 0))/d
        if (current_sum >= np.sum(max_average_mounth)):
            max_average_mounth = np.sum(Z[i:i+d, :, :], 0)/d
        if (current_sum <= np.sum(min_average_mounth)):
            min_average_mounth = np.sum(Z[i:i+d, :, :], 0)/d

        current_sum = np.sum(np.sum(Zl[i:i+d, :, 0], 0))/d
        if (current_sum >= np.sum(max_last_year_mounth)):
            max_last_year_mounth = np.sum(Zl[i:i+d, :, :], 0)/d
        if (current_sum <= np.sum(min_last_year_mounth)):
            min_last_year_mounth = np.sum(Zl[i:i+d, :, :], 0)/d

        i += d

    """ print(perf_counter()-t1)
    print(sum(max_average_mounth), str(max_average_mounth))
    print(sum(min_average_mounth), str(min_average_mounth))
    print(sum(max_last_year_mounth), str(max_last_year_mounth))
    print(sum(min_last_year_mounth), str(min_last_year_mounth)) """

    for i in range(4):
        av_d = [max_average_mounth, min_average_mounth, max_last_year_mounth, min_last_year_mounth][i]
        day_radiation_plot(av_d, orig_coord, data, city_plot_folder, i)

def gds_generation_curve(sts:list[str] = [], geocodes:list[str] = []) -> None:
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
                
                p.starmap(curves_gen, [[line[1:-2].split('";"'), orig_coord, closest_timeseries] for (line, orig_coord, closest_timeseries) in zip(lines[:1], city_ventures_coords, city_coords[idxs])])

    print(perf_counter()-t0)

if __name__ == "__main__":
    gds_generation_curve(geocodes=['3501608'])

# (0, '"CodEmpreendimento') (1, 'CodMunicipioIbge') (2, 'NumCoordNEmpreendimento') (3, 'NumCoordEEmpreendimento')
# (4, 'MdaPotenciaInstaladaKW') (5, 'MdaAreaArranjo') (6, 'QtdModulos') (7, 'MdaPotenciaModulos') (8, 'NomModeloModulo')
# (9, 'NomFabricanteModulo') (10, 'MdaPotenciaInversores') (11, 'NomModeloInversor"') (12, 'NomFabricanteInversor')
# (13, 'QtdUCRecebeCredito') (14, 'CodClasseConsumo') (15, 'DscSubGrupoTarifario') (16, 'SigModalidadeEmpreendimento')
# (17, 'SigTipoConsumidor') (18, 'NumCPFCNPJ') (19, 'CodCEP') (20, 'NumCNPJDistribuidora') (21, 'NomAgente')
# (22, 'NomSubEstacao') (23, 'NumCoordNSub') (24, 'NumCoordESub') (25, 'SigTipoGeracao') (26, 'DscPorte') (27, 'DthAtualizaCadastralEmpreend"\n')