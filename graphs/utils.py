import pickle
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

DATA_DIR = Path("data_vci")
FIGURES_DIR = Path("vci_figures")

map_size = {
    "1": "1B",
    "2": "2B",
    "4": "4B",
    "8": "8B",
    "16": "16B",
    "32": "32B",
    "64": "64B",
    "128": "128B",
    "256": "256B",
    "512": "512B",
    "1024": "1KB",
    "2048": "2KB",
    "4096": "4KB",
    "8192": "8KB",
    "16384": "16KB",
    "32768": "32KB",
    "65536": "64KB",
    "131072": "128KB",
    "262144": "256KB",
    "524288": "512KB",
    "1048576": "1MB",
    "2097152": "2MB",
    "4194304": "4MB",
    "8388608": "8MB",
    "16777216": "16MB",
    "33554432": "32MB",
    "67108864": "64MB",
    "134217728": "128MB",
    "268435456": "256MB",
    "536870912": "512MB",
    "1073741824": "1GB",
}

# Configurando o tamanho das fontes globalmente
plt.rcParams.update({
    'font.size': 12,          # Tamanho padrão da fonte
    'axes.titleweight': 'normal',  # Peso da fonte no título do gráfico
    'axes.titlesize': 12,     # Tamanho do título do gráfico
    'axes.labelsize': 12,     # Tamanho dos rótulos dos eixos
    'xtick.labelsize': 10,    # Tamanho das etiquetas no eixo X
    'ytick.labelsize': 10,    # Tamanho das etiquetas no eixo Y
    'legend.fontsize': 10,    # Tamanho da fonte na legenda
    'legend.title_fontsize': 12  # Tamanho do título da legenda
})

def load_data(bench_data: Path) -> tuple[pd.DataFrame, dict]:
    try:
        with bench_data.open("rb") as data:
            bench_dict = pickle.load(data)
            metadata = bench_dict["metadata"]
            df = pd.DataFrame(bench_dict["dataframe"]).T.drop_duplicates().T
            df.rename(columns={'lookups': 'lookup_size'}, inplace=True)
            return df, metadata
    except Exception as e:
        print(f"Error trying to open file {bench_data}: {e}")

def aggregate_data(df, by_columns:list[str]):
    # Group by 'Lookups' and calculate mean and std for 'Runtime'
    aggregated_df = df.groupby(by_columns).agg(
        runtime_mean=('runtime', 'mean'),
        runtime_std=('runtime', 'std'),
    ).reset_index()
    
    return aggregated_df

def aggregate_data_mb(df):
    # Group by 'Lookups' and calculate mean and std for 'Runtime'
    aggregated_df = df.groupby(['N', 'NumDevices']).agg(
        Runtime_mean=('Runtime', 'mean'),
        Runtime_std=('Runtime', 'std'),
    ).reset_index()
    
    return aggregated_df

@lru_cache(maxsize=2)
def generate_byte_mapping(start, end):
    byte_units = ['B', 'KB', 'MB', 'GB']
    mapping = {}

    # Gera potências de 2 de 1B até 128GB (2^37 bytes)
    for exp in range(start, end+1):
        value_in_bytes = 2 ** exp
        
        if value_in_bytes < 1024:
            # 1B até 1023B
            mapping[str(value_in_bytes)] = f"{value_in_bytes}B"
        elif value_in_bytes < 1024 ** 2:
            # 1KB até 1023KB
            value_in_kb = value_in_bytes // 1024
            mapping[str(value_in_bytes)] = f"{value_in_kb}KB"
        elif value_in_bytes < 1024 ** 3:
            # 1MB até 1023MB
            value_in_mb = value_in_bytes // (1024 ** 2)
            mapping[str(value_in_bytes)] = f"{value_in_mb}MB"
        else:
            # 1GB até 128GB
            value_in_gb = value_in_bytes // (1024 ** 3)
            mapping[str(value_in_bytes)] = f"{value_in_gb}GB"
    
    return mapping


def find_metg(group, x_data='task_granularity', y_data='efficiency', method:int = 1):
    group = group.sort_values(x_data)
    x = group[x_data].astype(float).values
    y = group[y_data].astype(float).values

    # Verifica se há cruzamento com 50%
    crosses_50 = (np.min(y) <= 50 <= np.max(y))

    if not crosses_50:
        return pd.Series({'metg': np.nan})


    if method == 1:
        # Método 1: interpolação cúbica y = f(x) + root-finding
        try:
            f = interp1d(x, y, kind='cubic', fill_value='extrapolate')

            def func(x_val):
                return float(f(x_val) - 50)

            for i in range(len(x) - 1):
                if (y[i] - 50) * (y[i + 1] - 50) < 0:
                    sol = root_scalar(func, bracket=[x[i], x[i + 1]], method='brentq')
                    if sol.converged:
                        return pd.Series({'metg': sol.root})
        except Exception as e:
            pass  # Falhou, tenta o fallback

    elif method == 2:
        # Método 2: interpolação linear invertida x = f(y)
        try:
            # Identifica dois pontos que cruzam 50%
            for i in range(len(y) - 1):
                y0, y1 = y[i], y[i + 1]
                if (y0 - 50) * (y1 - 50) < 0:
                    x0, x1 = x[i], x[i + 1]
                    # Interpolação linear entre os dois
                    metg = x0 + (50 - y0) * (x1 - x0) / (y1 - y0)
                    return pd.Series({'metg': float(metg)})
        except Exception as e:
            pass

    return pd.Series({'metg': np.nan})

@dataclass
class PlotMeta:
    x_axis: str
    y_axis: str
    hue: str
    style: str
    xlabel: str
    ylabel: str
    xticks: list
    yticks: list
    legend_title: str
    legend_labels: list


def generic_line_plot(dataset,
    title,
    output_file,
    plot_meta: PlotMeta,
    dashes: bool=False,
    markers=True,
    palette='tab10',
    size=None,
    sizes=None,
    xticks_rotation='horizontal'
):
    # Convert text width to inches (1 inch ≈ 2.54 cm)
    text_width_cm = 12  # Springer LNCS default
    width_inch = text_width_cm / 2.54  # ~4.72 inches
    height_inch = width_inch / 1.6  # ~2.95 inches (adjust ratio as needed)
    plt.figure(figsize=(width_inch, height_inch))

    line_plot = sns.lineplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        style=plot_meta.style,
        size=size,
        sizes=sizes,
        dashes=dashes,
        markers=markers,
        palette=palette,
    )

    line_plot.set_title(title)
    line_plot.set_xlabel(plot_meta.xlabel)
    line_plot.set_ylabel(plot_meta.ylabel)
    if len(plot_meta.xticks) > 0:
        line_plot.set_xticks(ticks=list(dataset[plot_meta.x_axis].unique()), labels=plot_meta.xticks, rotation=xticks_rotation)
    if len(plot_meta.yticks) > 0:
        line_plot.set_yticks(ticks=plot_meta.yticks)

    line_plot.grid(visible=True, linestyle='--')

    # Personalizar a legenda com labels customizadas
    handles, labels = line_plot.get_legend_handles_labels()
    if len(plot_meta.legend_labels) > 0:
        labels = plot_meta.legend_labels

    if len(plot_meta.legend_title):
        plt.legend(handles=handles, labels=labels, title=plot_meta.legend_title)

    plt.tight_layout()

    if (isinstance(output_file, Path)):
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi = 1200)
    else:
        plt.savefig(FIGURES_DIR/output_file, bbox_inches='tight', pad_inches=0, dpi = 1200)

def generic_line_plot_err_bar(
    dataset,
    title,
    output_file,
    plot_meta: PlotMeta,
    dashes: bool=False,
    markers=True,
    xticks_rotation='horizontal',
    outside_legend=False,
):
    # Convert text width to inches (1 inch ≈ 2.54 cm)
    text_width_cm = 12  # Springer LNCS default
    width_inch = text_width_cm / 2.54  # ~4.72 inches
    height_inch = width_inch / 1.6  # ~2.95 inches (adjust ratio as needed)
    # plt.figure(figsize=(width_inch, height_inch))
    plt.figure(figsize=(6, 4))

    colors = ["#8A57A5", "#377612"]

    line_plot = sns.lineplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        style=plot_meta.style,
        dashes=dashes,
        markers=markers,
        palette=colors if len(dataset[plot_meta.hue].unique()) == 2 else 'tab10',
        err_style="band",
        errorbar="ci",
        n_boot=1000,
        # err_kws={'elinewidth': 1.5, 'capsize': 5},
    )

    line_plot.set_title(title)
    line_plot.set_xlabel(plot_meta.xlabel)
    line_plot.set_ylabel(plot_meta.ylabel)
    if len(plot_meta.xticks) > 0:
        line_plot.set_xticks(ticks=list(dataset[plot_meta.x_axis].unique()), labels=plot_meta.xticks, rotation=xticks_rotation)
    if len(plot_meta.yticks) > 0:
        line_plot.set_yticks(ticks=plot_meta.yticks)

    line_plot.grid(visible=True, linestyle='--')

    # line_plot.set_xscale('log')

    # Personalizar a legenda com labels customizadas
    # handles, labels = line_plot.get_legend_handles_labels()
    # if len(plot_meta.legend_labels) > 0:
    #     labels = plot_meta.legend_labels

    # if len(plot_meta.legend_title):
    #     plt.legend(handles=handles, labels=labels, title=plot_meta.legend_title)

    # plt.tight_layout()

    # if (isinstance(output_file, Path)):
    #     plt.savefig(output_file, format='png', bbox_inches='tight', pad_inches=0, dpi = 1200)
    # else:
    #     plt.savefig(FIGURES_DIR/output_file, format='png', bbox_inches='tight', pad_inches=0, dpi = 1200)

    # Personalizar a legenda com labels customizadas
    # handles, labels = line_plot.get_legend_handles_labels()

    handles = line_plot.lines

    combos = dataset[[plot_meta.hue, plot_meta.style]].drop_duplicates()
    
    # Cria as labels evitando a duplicação se o hue e style forem o mesmo valor
    labels = []
    for hue_val, style_val in combos.to_numpy():
        if hue_val == style_val:
            labels.append(str(hue_val))
        else:
            labels.append(f"{hue_val}-{style_val}")

    # Remove seaborn's legend
    line_plot.legend_.remove()

    # if len(plot_meta.legend_labels) > 0:
    #     labels = plot_meta.legend_labels

    # order = sorted(
    #     zip(handles, labels),
    #     key=lambda x: x[0].get_linestyle() != '-'  # sólidos primeiro
    # )

    # handles, labels = zip(*order)

    # if len(plot_meta.legend_title):
    if outside_legend:
        plt.legend(
            handles=handles[:len(labels)],
            labels=labels,
            title=plot_meta.legend_title,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),  # center above the axes
            ncol=len(labels),
            frameon=True,
            borderaxespad=0.
        )
    else:
        plt.legend(
            handles=handles[:len(labels)],
            labels=labels,
            title=plot_meta.legend_title
        )

    plt.tight_layout()

    if (isinstance(output_file, Path)):
        plt.savefig(output_file, bbox_inches='tight', dpi=1200)
    else:
        plt.savefig(FIGURES_DIR/output_file, bbox_inches='tight', dpi=1200)


def generic_line_plot_err_bar_2(
    dataset,
    title,
    output_file,
    plot_meta, 
    dashes: bool = False,
    markers = True,
    xticks_rotation = 'horizontal',
    fifty_percent_line: bool = False,
    log_scale: bool = False,
    log_scale_y: bool = False,
    outside_legend: bool = False,
    bicolor_system: bool = False, 
    bicolor_palette: list = None, # <--- NEW OPTIONAL PARAMETER
):
    plt.figure(figsize=(6, 4))

    hue_vals = dataset[plot_meta.hue].unique() if plot_meta.hue else []
    unique_hues = len(hue_vals)

    # =========================================================
    # Bicolor System & Marker Type Grouping
    # =========================================================
    if bicolor_system and plot_meta.hue and plot_meta.style:
        sys_parts = sorted(list(set([val.split(' - ')[0] for val in hue_vals if ' - ' in val])))
        type_parts = sorted(list(set([val.split(' - ')[1] for val in hue_vals if ' - ' in val])))
        
        # Use passed palette or default to your preferred colors
        if bicolor_palette is None:
            base_colors = ["#8A57A5", "#377612"]
        else:
            base_colors = bicolor_palette
            
        base_markers = ['o', 's', '^', 'D', 'v', 'p', 'X']
        base_dashes = [(2, 2), (5, 5), (1, 1), (3, 1, 1, 1)] 
        
        custom_palette = {}
        custom_markers = {}
        custom_dashes = {}
        
        for val in hue_vals:
            s_part = val.split(' - ')[0] if ' - ' in val else val
            t_part = val.split(' - ')[1] if ' - ' in val else val
            
            # Map System to Color (modulo to avoid index errors if more than 2 systems appear)
            c_idx = sys_parts.index(s_part) if s_part in sys_parts else 0
            custom_palette[val] = base_colors[c_idx % len(base_colors)]
            
            # Map Type to Marker and Dash
            m_idx = type_parts.index(t_part) if t_part in type_parts else 0
            custom_markers[val] = base_markers[m_idx % len(base_markers)]
            custom_dashes[val] = base_dashes[m_idx % len(base_dashes)]

        palette_arg = custom_palette
        markers_arg = custom_markers if markers else False
        dashes_arg = custom_dashes if dashes else False

    else:
        # Original logic fallback
        if unique_hues <= 2:
            palette_arg = ["#8A57A5", "#377612"][:max(1, unique_hues)]
        else:
            palette_arg = sns.color_palette("tab10", max(1, unique_hues))
            
        markers_arg = markers
        dashes_arg = dashes

    line_plot = sns.lineplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        style=plot_meta.style,
        dashes=dashes_arg,
        markers=markers_arg,
        palette=palette_arg if plot_meta.hue is not None else None,
        err_style="band",
        errorbar="ci",
        n_boot=1000,
    )

    line_plot.set_title(title)
    line_plot.set_xlabel(plot_meta.xlabel)
    line_plot.set_ylabel(plot_meta.ylabel)
    
    if len(plot_meta.xticks) > 0:
        line_plot.set_xticks(ticks=list(dataset[plot_meta.x_axis].unique()), labels=plot_meta.xticks, rotation=xticks_rotation)
    if len(plot_meta.yticks) > 0:
        line_plot.set_ylim(plot_meta.yticks[0], plot_meta.yticks[-1])
        line_plot.set_yticks(ticks=plot_meta.yticks)

    if fifty_percent_line:
        ymax = line_plot.get_ylim()[1]
        half_ymax = 50
        line_plot.axhline(y=half_ymax, color='red', linestyle='--')

    if log_scale:
        line_plot.set_xscale('log')
        line_plot.invert_xaxis()

    if log_scale_y:
        import matplotlib.ticker as ticker
        line_plot.set_yscale('log')
        line_plot.yaxis.set_major_locator(ticker.LogLocator(base=10))
        line_plot.yaxis.set_minor_locator(ticker.NullLocator())

    line_plot.grid(visible=True, linestyle='--')

    handles, labels = line_plot.get_legend_handles_labels()
    
    if labels and labels[0] == plot_meta.hue:
        handles = handles[1:]
        labels = labels[1:]

    if plot_meta.legend_labels and len(plot_meta.legend_labels) == len(labels):
        labels = plot_meta.legend_labels

    if outside_legend:
        plt.legend(
            handles=handles,
            labels=labels,
            title=plot_meta.legend_title,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=9,                
            title_fontsize=11,
            frameon=True,
            borderaxespad=0.
        )
    else:
        plt.legend(
            handles=handles,
            labels=labels,
            title=plot_meta.legend_title
        )

    plt.tight_layout()

    if isinstance(output_file, Path):
        plt.savefig(output_file, bbox_inches='tight', dpi=1200)
    else:
        plt.savefig(output_file, bbox_inches='tight', dpi=1200)

def generic_line_plot_err_bar_3(
    dataset,
    title,
    output_file,
    plot_meta,
    dashes: bool = False,
    markers=True,
    xticks_rotation="horizontal",
    fifty_percent_line: bool = False,
    log_scale: bool = False,
    log_scale_y: bool = False,
    outside_legend: bool = False,
    tricolor_system: bool = False,
    tricolor_palette: list = None,
):
    plt.figure(figsize=(6, 4))

    hue_vals = dataset[plot_meta.hue].unique() if plot_meta.hue else []
    unique_hues = len(hue_vals)

    # =========================================================
    # Tricolor System & Marker Type Grouping
    # =========================================================
    if tricolor_system and plot_meta.hue and plot_meta.style:
        sys_parts = sorted(
            list(set([val.split(" - ")[0] for val in hue_vals if " - " in val]))
        )
        type_parts = sorted(
            list(set([val.split(" - ")[1] for val in hue_vals if " - " in val]))
        )

        # Default 3-color palette (Purple, Green, Orange) or use custom passed palette
        if tricolor_palette is None:
            base_colors = ["#8A57A5", "#377612", "#D95F02"]
        else:
            base_colors = tricolor_palette

        base_markers = ["o", "s", "^", "D", "v", "p", "X"]
        base_dashes = [(2, 2), (5, 5), (1, 1), (3, 1, 1, 1)]

        custom_palette = {}
        custom_markers = {}
        custom_dashes = {}

        for val in hue_vals:
            s_part = val.split(" - ")[0] if " - " in val else val
            t_part = val.split(" - ")[1] if " - " in val else val

            # Map System to Color (modulo handles cases where > 3 systems exist)
            c_idx = sys_parts.index(s_part) if s_part in sys_parts else 0
            custom_palette[val] = base_colors[c_idx % len(base_colors)]

            # Map Type to Marker and Dash pattern
            m_idx = type_parts.index(t_part) if t_part in type_parts else 0
            custom_markers[val] = base_markers[m_idx % len(base_markers)]
            custom_dashes[val] = base_dashes[m_idx % len(base_dashes)]

        palette_arg = custom_palette
        markers_arg = custom_markers if markers else False
        dashes_arg = custom_dashes if dashes else False

    else:
        # Fallback logic for general plots
        default_tricolor = ["#8A57A5", "#377612", "#D95F02"]
        if unique_hues <= 3:
            palette_arg = default_tricolor[: max(1, unique_hues)]
        else:
            palette_arg = sns.color_palette("tab10", max(1, unique_hues))

        markers_arg = markers
        dashes_arg = dashes

    line_plot = sns.lineplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        style=plot_meta.style,
        dashes=dashes_arg,
        markers=markers_arg,
        palette=palette_arg if plot_meta.hue is not None else None,
        err_style="band",
        errorbar="ci",
        n_boot=1000,
    )

    line_plot.set_title(title)
    line_plot.set_xlabel(plot_meta.xlabel)
    line_plot.set_ylabel(plot_meta.ylabel)

    if len(plot_meta.xticks) > 0:
        line_plot.set_xticks(
            ticks=list(dataset[plot_meta.x_axis].unique()),
            labels=plot_meta.xticks,
            rotation=xticks_rotation,
        )
    if len(plot_meta.yticks) > 0:
        line_plot.set_ylim(plot_meta.yticks[0], plot_meta.yticks[-1])
        line_plot.set_yticks(ticks=plot_meta.yticks)

    if fifty_percent_line:
        half_ymax = 50
        line_plot.axhline(y=half_ymax, color="red", linestyle="--")

    if log_scale:
        line_plot.set_xscale("log")
        line_plot.invert_xaxis()

    if log_scale_y:
        line_plot.set_yscale("log")
        line_plot.yaxis.set_major_locator(ticker.LogLocator(base=10))
        line_plot.yaxis.set_minor_locator(ticker.NullLocator())

    line_plot.grid(visible=True, linestyle="--")

    handles, labels = line_plot.get_legend_handles_labels()

    if labels and labels[0] == plot_meta.hue:
        handles = handles[1:]
        labels = labels[1:]

    if plot_meta.legend_labels and len(plot_meta.legend_labels) == len(labels):
        labels = plot_meta.legend_labels

    if outside_legend:
        plt.legend(
            handles=handles,
            labels=labels,
            title=plot_meta.legend_title,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,  # Adjusted to 3 columns for tricolor systems
            fontsize=9,
            title_fontsize=11,
            frameon=True,
            borderaxespad=0.0,
        )
    else:
        plt.legend(handles=handles, labels=labels, title=plot_meta.legend_title)

    plt.tight_layout()

    plt.savefig(output_file, bbox_inches="tight", dpi=1200)

def generic_bar_plot(dataset, title, output_file, plot_meta: PlotMeta):
    # Criar o gráfico de linhas
    plt.figure(figsize=(10, 6))
    bar_plot = sns.barplot(
        data=dataset,
        x=plot_meta.x_axis,
        y=plot_meta.y_axis,
        hue=plot_meta.hue,
        # style=plot_meta.style,
        palette='colorblind',
        errorbar=None,
        edgecolor='black'
    )

    hatches = ['x', '*', '+', 'o']

    rectangles = [patch for patch in bar_plot.patches if isinstance(patch, Rectangle)]

    num_bars_per_group = len(dataset['NumGPUs'].unique())
    # Itera e aplica hatches
    for i, thisbar in enumerate(rectangles):
        if (i >= len(dataset)):
            break
        thisbar.set_edgecolor('black')
        thisbar.set_linewidth(1.5)
        # Itera sobre o tamanho de grupos únicos no hue (Space Order)
        thisbar.set_hatch(hatches[i // num_bars_per_group])

    # Corrige a legenda com hatches corretos
    handles, labels = bar_plot.get_legend_handles_labels()
    for handle, hatch in zip(handles, hatches):
        handle.set_hatch(hatch)
    bar_plot.legend_.remove()


    bar_plot.grid(visible=True, linestyle='--')
    bar_plot.set_title(title)
    bar_plot.set_xlabel(plot_meta.xlabel)
    bar_plot.set_ylabel(plot_meta.ylabel)
    # bar_plot.set_xticks(ticks=plot_meta.xticks)
    if len(plot_meta.yticks) > 0:
        bar_plot.set_yticks(ticks=plot_meta.yticks)

    legend = plt.legend(handles=handles, labels=labels, title=plot_meta.legend_title)
    plt.setp(legend.get_title(), fontweight='bold')
    plt.savefig(FIGURES_DIR/output_file, bbox_inches='tight')


def metg_relative_bar_subplots(
    dataset,
    output_file,
    runtimes=('mpi_openmp', 'mpp', 'starpu'),
    patterns=('trivial', 'tree', 'stencil_1d'),
    num_devices=(4, 16, 32),
    output_value=16,
):
    """Plot METG relative to the trivial pattern for each runtime.

    The input is the ``metg_df`` produced by the weak-scaling analysis. Each
    pattern is divided by the METG of ``trivial`` for the same runtime,
    device count, and output size.
    """
    required_columns = {'name', 'num_devices', 'type', 'metg'}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    data = dataset.copy()
    if output_value is not None and 'output' in data.columns:
        data = data[data['output'] == output_value].copy()

    data = data[
        data['name'].isin(runtimes)
        & data['type'].isin(patterns)
        & data['num_devices'].isin(num_devices)
    ].copy()

    key_columns = ['name', 'num_devices']
    if 'output' in data.columns:
        key_columns.append('output')

    trivial = data[data['type'] == 'trivial'][key_columns + ['metg']].rename(
        columns={'metg': 'trivial_metg'}
    )
    if trivial.empty:
        raise ValueError('No trivial METG values available for normalization')

    data = data.merge(trivial, on=key_columns, how='inner', validate='many_to_one')
    data['relative_metg'] = data['metg'] / data['trivial_metg']

    pattern_labels = {
        'trivial': 'Trivial',
        'tree': 'Tree',
        'stencil_1d': 'Stencil',
    }
    runtime_labels = {
        'mpi_openmp': 'MPI+OpenMP',
        'mpp': 'MPP',
        'starpu': 'StarPU',
    }
    palette = {
        'trivial': '#8A57A5',
        'tree': '#377612',
        'stencil_1d': '#D95F02',
    }
    hatches = {
        'trivial': '',
        'tree': '//',
        'stencil_1d': '..',
    }

    fig, axes = plt.subplots(
        1,
        len(runtimes),
        figsize=(9, 3.5),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    legend_handles = [
        Patch(
            facecolor=palette[pattern],
            edgecolor='black',
            hatch=hatches[pattern],
        )
        for pattern in patterns
    ]
    legend_labels = [pattern_labels.get(pattern, pattern) for pattern in patterns]

    for ax, runtime in zip(axes, runtimes):
        runtime_data = data[data['name'] == runtime].copy()
        runtime_data['num_devices'] = pd.Categorical(
            runtime_data['num_devices'], categories=num_devices, ordered=True
        )
        runtime_data['type'] = pd.Categorical(
            runtime_data['type'], categories=patterns, ordered=True
        )

        bar_plot = sns.barplot(
            data=runtime_data,
            x='num_devices',
            y='relative_metg',
            hue='type',
            order=num_devices,
            hue_order=patterns,
            palette=palette,
            errorbar=None,
            edgecolor='black',
            ax=ax,
        )

        for pattern, container in zip(patterns, bar_plot.containers):
            for rectangle in container:
                rectangle.set_linewidth(0.8)
                rectangle.set_hatch(hatches[pattern])

        if ax.legend_ is not None:
            ax.legend_.remove()

        ax.axhline(1.0, color='black', linestyle='--', linewidth=0.9, zorder=0)
        ax.set_title(runtime_labels.get(runtime, runtime))
        ax.set_xlabel('Number of GPUs')
        ax.set_xticks(range(len(num_devices)), labels=[str(device) for device in num_devices])
        ax.grid(axis='y', visible=True, linestyle='--', linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('METG / METG(trivial)')
    for ax in axes[1:]:
        ax.set_ylabel('')

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(patterns),
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_file, bbox_inches='tight', dpi=1200)
    plt.close(fig)


def metg_relative_bar_subplots_2(
    dataset,
    output_file,
    runtimes=('mpi_openmp', 'mpp', 'starpu'),
    patterns=('trivial', 'tree', 'stencil_1d'),
    num_devices=(4, 16, 32),
    output_value=16,
):
    """Plot METG relative to the trivial pattern for each runtime.

    The input is the ``metg_df`` produced by the weak-scaling analysis. Each
    pattern is divided by the METG of ``trivial`` for the same runtime,
    device count, and output size.

    The trivial pattern is used for normalization but is not plotted because
    its relative METG is always 1.
    """
    required_columns = {'name', 'num_devices', 'type', 'metg'}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    data = dataset.copy()

    if output_value is not None and 'output' in data.columns:
        data = data[data['output'] == output_value].copy()

    data = data[
        data['name'].isin(runtimes)
        & data['type'].isin(patterns)
        & data['num_devices'].isin(num_devices)
    ].copy()

    # Keep 'trivial' for normalization.
    key_columns = ['name', 'num_devices']
    if 'output' in data.columns:
        key_columns.append('output')

    trivial = data[data['type'] == 'trivial'][key_columns + ['metg']].rename(
        columns={'metg': 'trivial_metg'}
    )

    if trivial.empty:
        raise ValueError('No trivial METG values available for normalization')

    data = data.merge(
        trivial,
        on=key_columns,
        how='inner',
        validate='many_to_one'
    )

    data['relative_metg'] = data['metg'] / data['trivial_metg']

    # Do not plot trivial since it is always equal to 1.
    plot_patterns = [pattern for pattern in patterns if pattern != 'trivial']

    pattern_labels = {
        'trivial': 'Trivial',
        'tree': 'Tree',
        'stencil_1d': 'Stencil',
    }

    runtime_labels = {
        'mpi_openmp': 'MPI+OpenMP',
        'mpp': 'MPP',
        'starpu': 'StarPU+OpenMP',
    }

    palette = {
        'trivial': '#8A57A5',
        'tree': '#377612',
        'stencil_1d': '#D95F02',
    }

    hatches = {
        'trivial': '',
        'tree': '//',
        'stencil_1d': '..',
    }

    fig, axes = plt.subplots(
        1,
        len(runtimes),
        figsize=(9, 3.5),
        sharey=True,
        squeeze=False,
    )

    axes = axes[0]

    # Legend only contains the patterns that are actually plotted.
    legend_handles = [
        Patch(
            facecolor=palette[pattern],
            edgecolor='black',
            hatch=hatches[pattern],
        )
        for pattern in plot_patterns
    ]

    legend_labels = [
        pattern_labels.get(pattern, pattern)
        for pattern in plot_patterns
    ]

    for ax, runtime in zip(axes, runtimes):
        runtime_data = data[
            (data['name'] == runtime)
            & (data['type'].isin(plot_patterns))
        ].copy()

        runtime_data['num_devices'] = pd.Categorical(
            runtime_data['num_devices'],
            categories=num_devices,
            ordered=True,
        )

        runtime_data['type'] = pd.Categorical(
            runtime_data['type'],
            categories=plot_patterns,
            ordered=True,
        )

        bar_plot = sns.barplot(
            data=runtime_data,
            x='num_devices',
            y='relative_metg',
            hue='type',
            order=num_devices,
            hue_order=plot_patterns,
            palette=palette,
            errorbar=None,
            edgecolor='black',
            ax=ax,
        )

        for pattern, container in zip(plot_patterns, bar_plot.containers):
            for rectangle in container:
                rectangle.set_linewidth(0.8)
                rectangle.set_hatch(hatches[pattern])

        if ax.legend_ is not None:
            ax.legend_.remove()

        # Reference line corresponding to the trivial METG.
        ax.axhline(
            1.0,
            color='black',
            linestyle='--',
            linewidth=0.9,
            zorder=0,
        )

        ax.set_title(runtime_labels.get(runtime, runtime))
        ax.set_xlabel('Number of GPUs')
        ax.set_xticks(
            range(len(num_devices)),
            labels=[str(device) for device in num_devices],
        )
        ax.grid(
            axis='y',
            visible=True,
            linestyle='--',
            linewidth=0.6,
        )
        ax.set_axisbelow(True)

    axes[0].set_ylabel('METG / METG(trivial)')

    for ax in axes[1:]:
        ax.set_ylabel('')

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(plot_patterns),
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_file, bbox_inches='tight', dpi=1200)
    plt.close(fig)

def nbody_roofline(
    hip_stats_file,
    hip_trace_file,
    openmp_stats_file,
    openmp_trace_file,
    output_file,
    n_bodies=2**20,
    num_gpus=1,
    peak_flops_per_gpu_tflops=22.6,
    peak_bandwidth_per_gpu_tbps=1.6,
    float_bytes=4,
    reuse_target_position=True,
    reuse_source_position=True,
):
    """Generate a roofline plot from HIP and OpenMP ROCprof profiles.

    The operation and traffic model follows the source kernels. ``bodyForce``
    performs 18 logical FLOPs per body interaction: 17 arithmetic operations
    plus one reciprocal-square-root operation. Its target position is counted
    once per target by default because it is loop invariant and can stay in
    registers. The source position array is counted once per dispatch by
    default because every target traverses the same array and the accesses can
    be served by the cache hierarchy. ``integratePositions`` performs six
    FLOPs per body.

    When ``num_gpus`` is one, the first GPU agent in each trace is selected.
    This makes the HIP and OpenMP points comparable as single-GPU results even
    if the OpenMP profile was collected with four agents.
    """
    profile_files = {
        'HIP': (Path(hip_stats_file), Path(hip_trace_file)),
        'OpenMP': (Path(openmp_stats_file), Path(openmp_trace_file)),
    }
    pos_bytes = 3 * float_bytes
    vel_bytes = 3 * float_bytes
    bodyforce_flops_per_interaction = 18
    bodyforce_flops_per_target = 6
    integrate_flops_per_body = 6
    peak_flops = peak_flops_per_gpu_tflops * num_gpus
    peak_bandwidth = peak_bandwidth_per_gpu_tbps * num_gpus
    records = []

    def kernel_kind(name):
        if 'bodyForce' in name:
            return 'bodyForce'
        if 'integratePositions' in name:
            return 'integratePositions'
        return None

    for implementation, (stats_file, trace_file) in profile_files.items():
        stats = pd.read_csv(stats_file)
        trace = pd.read_csv(trace_file)
        stats['kernel'] = stats['Name'].map(kernel_kind)
        trace['kernel'] = trace['Kernel_Name'].map(kernel_kind)
        stats = stats.dropna(subset=['kernel'])
        trace = trace.dropna(subset=['kernel']).copy()

        if trace.empty:
            raise ValueError(f'No recognized kernels found in {trace_file}')

        trace['Start_Timestamp'] = pd.to_numeric(trace['Start_Timestamp'])
        trace['End_Timestamp'] = pd.to_numeric(trace['End_Timestamp'])
        trace['Grid_Size_X'] = pd.to_numeric(trace['Grid_Size_X'])

        agents = sorted(trace['Agent_Id'].unique())
        selected_agents = agents[:num_gpus]
        trace = trace[trace['Agent_Id'].isin(selected_agents)].copy()

        for kernel in ('bodyForce', 'integratePositions'):
            kernel_trace = trace[trace['kernel'] == kernel].copy()
            if kernel_trace.empty:
                continue

            stats_row = stats[stats['kernel'] == kernel]
            if not stats_row.empty:
                expected_calls = int(stats_row['Calls'].sum())
                if expected_calls < len(kernel_trace):
                    raise ValueError(
                        f'{implementation} {kernel}: statistics report '
                        f'{expected_calls} calls, selected trace contains '
                        f'{len(kernel_trace)}'
                    )

            targets = kernel_trace['Grid_Size_X'].astype(float).to_numpy()
            if kernel == 'bodyForce':
                interactions = targets * n_bodies
                flops = (
                    interactions * bodyforce_flops_per_interaction
                    + targets * bodyforce_flops_per_target
                )
                target_position_traffic = (
                    targets * pos_bytes
                    if reuse_target_position
                    else interactions * pos_bytes
                )
                source_position_traffic = (
                    n_bodies * pos_bytes
                    if reuse_source_position
                    else interactions * pos_bytes
                )
                bytes_moved = (
                    source_position_traffic
                    + target_position_traffic
                    + targets * 2 * vel_bytes
                )
            else:
                flops = targets * integrate_flops_per_body
                bytes_moved = targets * (pos_bytes + vel_bytes + pos_bytes)

            kernel_time = (
                kernel_trace['End_Timestamp']
                - kernel_trace['Start_Timestamp']
            ).sum() / 1e9
            if kernel_time <= 0:
                raise ValueError(f'Invalid timestamp span for {implementation} {kernel}')

            records.append({
                'implementation': implementation,
                'kernel': kernel,
                'arithmetic_intensity': flops.sum() / bytes_moved.sum(),
                'performance_tflops': flops.sum() / kernel_time / 1e12,
                'calls': len(kernel_trace),
                'active_gpus': kernel_trace['Agent_Id'].nunique(),
                'kernel_time_s': kernel_time,
            })

    points = pd.DataFrame(records)
    if points.empty:
        raise ValueError('No roofline points were generated')

    colors = {'HIP': '#8A57A5', 'OpenMP': '#377612'}
    markers = {'bodyForce': 'o', 'integratePositions': 's'}
    x_min = max(points['arithmetic_intensity'].min() / 2, 1e-3)
    x_max = max(points['arithmetic_intensity'].max() * 10, 100)
    x_values = np.logspace(np.log10(x_min), np.log10(x_max), 400)
    memory_roof = peak_bandwidth * x_values
    compute_roof = np.full_like(x_values, peak_flops)
    ridge_point = peak_flops / peak_bandwidth

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(
        x_values,
        np.minimum(memory_roof, compute_roof),
        color='black',
        linewidth=1.5,
        label=f'MI210 x{num_gpus} roofline',
    )
    ax.loglog(
        x_values,
        memory_roof,
        color='#777777',
        linestyle=':',
        linewidth=0.8,
    )
    ax.axvline(ridge_point, color='#777777', linestyle=':', linewidth=0.8)

    for point in points.itertuples(index=False):
        label = (
            f'{point.implementation} {point.kernel} '
            f'({point.active_gpus} GPU' + ('s' if point.active_gpus != 1 else '') + ')'
        )
        ax.scatter(
            point.arithmetic_intensity,
            point.performance_tflops,
            color=colors[point.implementation],
            marker=markers[point.kernel],
            s=75,
            edgecolor='black',
            linewidth=0.7,
            zorder=3,
            label=label,
        )

    point_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[kernel],
            color='white',
            markerfacecolor=colors[implementation],
            markeredgecolor='black',
            markersize=7,
            linestyle='None',
            label=(
                f'{implementation} {kernel} '
                f'({int(points.query("implementation == @implementation and kernel == @kernel").iloc[0].active_gpus)} GPU)'
            ),
        )
        for implementation in ('HIP', 'OpenMP')
        for kernel in ('bodyForce', 'integratePositions')
        if not points.query("implementation == @implementation and kernel == @kernel").empty
    ]
    roof_handle = Line2D([0], [0], color='black', linewidth=1.5, label=f'MI210 x{num_gpus} roofline')
    ax.legend(handles=point_handles + [roof_handle], loc='lower right', fontsize=8)
    ax.set_xlabel('Arithmetic intensity (FLOP/byte)')
    ax.set_ylabel('Performance (TFLOP/s)')
    ax.set_title(f'N-Body roofline ({n_bodies:,} bodies)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(points['performance_tflops'].min() / 3, 1e-4), peak_flops * 1.25)
    ax.annotate(
        f'Ridge point: {ridge_point:.1f} FLOP/byte',
        xy=(ridge_point, peak_flops),
        xytext=(8, -20),
        textcoords='offset points',
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches='tight', dpi=1200)
    plt.close(fig)

    return points
