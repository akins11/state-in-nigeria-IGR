from pandas import DataFrame, value_counts
from numpy import where

from plotnine import ggplot, aes, position_dodge, labs
from plotnine import geom_col, geom_text, geom_segment, geom_point
from plotnine import scale_y_continuous, scale_x_discrete, scale_y_discrete, coord_flip
from plotnine import theme_538, theme, element_text
 
import matplotlib.pyplot as plt
import squarify as sq
    
# [] Cleaning the data frame ======================================================================================
def clean_data(df):
    """
    df : [Data Frame] a data frame to clean
    """

    # Creating Geo-political zones -----------------------------------------------------------------
    north_central= ["Benue", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau", "FCT"]
    north_east   = ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"]
    north_west   = ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"]
    south_east   = ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"]
    south_south  = ["Akwa Ibom", "Bayelsa", "Cross River", "Edo", "Rivers", "Delta"]
    south_west   = ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"]
    
    df.loc[df["State"].isin(north_central), "Zone"] = "North Central"
    df.loc[df["State"].isin(north_east), "Zone"]  = "North East"
    df.loc[df["State"].isin(north_west), "Zone"]  = "North West"
    df.loc[df["State"].isin(south_east), "Zone"]  = "South East"
    df.loc[df["State"].isin(south_south), "Zone"] = "South South"
    df.loc[df["State"].isin(south_west), "Zone"]  = "South West"
    
    # Relocate the zone column 
    col = df.pop("Zone")
    data = df.insert(1, col.name, col)
    
    return df


# [] format number ===================================================================================================
def comma_format(number, deci = None):
    if deci == None:
        result = "{:,}".format(number)
    else:
        round_number = round(number, deci)
        result = "{:,}".format(round_number)
    return result



# [] Reordering axis text using plotnine =============================================================================
def create_ordered_text(data, cat_col, num_col = None, sort_value = False, f_sort = True, rev = False, axis = "x"):
    """
    Parameter
    ---------
    data:       [data.frame] data to use in creating the ordered list.
    cat_col:    [object, category] a columns from the data to order.
    num_col:    [int64 float64] a column from the data to sort by, only when `summary` = True.
    sort_value: [bool] True to sort_values by `num_col` in the data, False to just count the no. distinct values in `cat_col`.
    f_sort:     [bool] True to sort or False not to sort.
    rev:        [bool] True to reverse the list, False not to.
    axis:       [string] The axis to apply the ordered text, 'x'-> x_axis, 'y'-> y_axis.
    
    Return
    ------
    if axis = 'x', <plotnine.scales.scale_xy.scale_x_discrete at [env]>
    if axis = 'y', <plotnine.scales.scale_xy.scale_y_discrete at [env]>
    """
    # checking for the right arguments -----------------------------------------------------------------------------
    if sort_value:
        if all(culs in data.columns.tolist() for culs in [cat_col, num_col]) != True:
            raise Exception(f"All or any of {[cat_col, num_col]} is not in the data")
        
        if (data[cat_col].dtype not in ["object", "category"]) or (data[num_col].dtype not in ["int64", "float64"]):
            raise Exception("argument (cat_col) requires an 'object' or a 'categorical' dtype and\n \
                             (num_col) requires an 'int64' or 'float64' dtype")
    else:
        if cat_col not in data.columns.tolist():
            raise Exception(f"variable {cat_col} is not in the data")
            
        if data[cat_col].dtype not in ["object", "category"]:
            raise Exception("Object argument `cat_col` requires an 'object' or a 'categorical' type of data")
            
    # A Summary df or not  -----------------------------------------------------------------------------------------
    if sort_value:
        rod_tbl = data.sort_values(num_col, ascending = False).reset_index(drop = True)
        ord_txt = rod_tbl[cat_col].tolist()
        
        if rev:
            ord_txt.reverse()
        else:
            ord_txt
    else:
        if rev:
            ord_txt = data[cat_col].value_counts(sort = f_sort).index.tolist()[::-1]
        else:
            ord_txt = data[cat_col].value_counts(sort = f_sort).index.tolist()
        
    # X_axis or y_axis ---------------------------------------------------------------------------------------------
    if axis == "x":
        return(scale_x_discrete(limits = ord_txt))
    elif axis == "y":
        return(scale_y_discrete(limits = ord_txt))
    else:
        raise ValueError(f"`axis` can only be either 'x' or 'y' and not {axis}")





# [] Revenue by States ================================================================================================
var_colr = {"Grand Total": "#00EE76",
            "Total Tax": "#00CD66",
            "MDAs": "#54FF9F",
            "PAYE": "#43CD80",
            "Direct Assessment": "#3CB371",
            "Road Tax": "#00FA9A"}

def get_revenue(df, by, revenue_var, period, colr = None, typ = "plt"):
    """
    df:          [DataFrame] for the analysis.
    by :         [object] get the revenue by state or by Zone.
    revenue_var: [float, int] revenue variable.
    period:      [str] the period of the analysis ("q1", "q2", "half year").
    colr:        [str] color of the plot bars.
    typ:         [str] analysis to generate 'plt' for plot 'tbl' for table.
    
    returns
    -------
    if typ = 'plt' : plot
    if typ = 'tbl' : dataFrame
    """
    
    # revenue data frame summary -------------------------------------------------------------
    if by == "State":
        f_tbl = df[[by, revenue_var]].sort_values(revenue_var, ascending = False)
        f_tbl["prop"] = round(f_tbl[revenue_var] / f_tbl[revenue_var].sum() * 100, 2)
        f_tbl.reset_index(drop = True, inplace = True)
        
    elif by == "Zone":
        f_tbl = DataFrame(df[[by, revenue_var]].groupby(by)[revenue_var].sum().reset_index())
        f_tbl["prop"] = round(f_tbl[revenue_var] / f_tbl[revenue_var].sum()*100, 2)
    else:
        raise ValueError(f"{by} is not a valid argument either 'State' or 'Zone'")
    
    # Proportion Text Position ---------------------------------------------------------------
    max_var_value = f_tbl[revenue_var].max()
    text_pos = max_var_value - (0.05*max_var_value)
    
    # colors ---------------------------------------------------------------------------------
    colr = var_colr[revenue_var] if colr is None else colr
    
    # labels & title -------------------------------------------------------------------------
    def title_l(using = by):
        period_dict = {"q1": "1st Quarter", "q2": "2nd Quarter", "half year": "First Half"}
        period_value = period_dict[period]
        
        if period == "half year":
            title_o = f"2021 {period_value} Total Revenue From {revenue_var} For Each {by}"
        elif period != "half year":
            title_o = f"Total Revenue Of {period_value} 2021 From {revenue_var} For Each {by}"
        else:
            raise ValueError(f"`period` {period} is not recognised")
        
        return title_o
    
    # Plot ----------------------------------------------------------------------------------
    f_plt = (
        ggplot(f_tbl, aes(by, revenue_var)) +
        geom_col(fill = colr) +
        geom_text(aes(label = "prop", y = text_pos), position = position_dodge(0.9), size = 9, ha = "left", format_string = "{}%") +
        coord_flip() +
        create_ordered_text(f_tbl, by, revenue_var, sort_value = True, axis = "x", rev = True) +
        scale_y_continuous(labels = lambda l: ["{:,.0f}".format(v) for v in l]) +
        labs(x = "", y = revenue_var, title = title_l()) +
        theme_538() +
        theme(figure_size = (9, 6))
    )
    
    if typ == "tbl":
        return f_tbl
    elif typ == "plt":
        return f_plt
    else:
        raise ValueError(f"argument `typ` can only be either 'tbl'-Table or 'plt'-Plot and not {tbl}")

        
    
    
# [] proportion table ==============================================================================================
def prop_table(df, var, deci = None):
    """
    df : [dataframe]
    var: [float, int] variable from the `df`.
    
    return
    ------
    the proportion of each value from the variable.
    """    
    # Condition ---------------------------------------------------------
    if deci is None:
        result = round(df[var] / df[var].sum()*100)
    else:
        result = round(df[var] / df[var].sum()*100, deci)
        
    return result



# [] source of state revenue =======================================================================================
def state_source(df, loc_var, period, colr = ["forestgreen", "darkgreen"], typ = "plt"):
    """
    Paremeters
    ----------
    df:      [dataframe]
    loc_var: [object] a location variable from the `df`.
    period:  [str] the period of the data e.i. ['q1', 'q2', 'half year'].
    colr:    [list $ str] the color to use in the plot.
    typ:     [str] analysis to generate 'plt' for plot 'tbl' for table.
    
    Returns
    -------
    if typ = 'plt' : plot
    if typ = 'tbl' : dataFrame
    """
    # Table ------------------------------------------------------------------------------------------------------------------------------------------
    f_tbl = (df[df["State"] == loc_var]
              .drop(labels = ["Grand Total", "Total Tax"], axis = 1)
              .melt(id_vars = ["State", "Zone"], value_vars = ["PAYE", "Direct Assessment", "Road Tax", "Other Taxes", "MDAs"], var_name = "revenue"))
    f_tbl["prop"] = prop_table(f_tbl, "value")
        
    # labels & title --------------------------------------------------------------------------------------------------------------------------------
    def title_l():
        period_dict = {"q1": "1st Quarter", "q2": "2nd Quater", "half year": "First Half"}
        try:
            period_value = period_dict[period]
            
            if period == "half year":
                if_tlt = f"{period_value} Source Of {loc_var} State Internally Generated Revenue"
                el_tlt = f"{period_value} Source Of {loc_var} Internally Generated Revenue"
                title_o = if_tlt if loc_var != "FCT" else el_tlt
            else:
                if_tlt = f"Source Of {loc_var} State Internally Generated Revenue For The {period_value} Of 2021"
                el_tlt = f"Source Of {loc_var} Internally Generated Revenue For The {period_value} Of 2021"
                title_o = if_tlt if loc_var != "FCT" else el_tlt
                
            return title_o
        except:
            raise ValueError(f"{period} in not a valid period name")
    
    # Plot -----------------------------------------------------------------------------------------------------------------------------------------
    f_plt = (
            ggplot(f_tbl, aes("reorder(revenue, value)", "value")) +
            geom_segment(aes(xend = "revenue", y = 0,  yend = "value"), color = colr[0]) +
            geom_point(size = 16, color = colr[1]) +
            geom_text(aes(label = "prop"), position = position_dodge(0.9), color = "white", size = 8, fontweight = "bold", format_string = "{}%") +
            coord_flip() +
            scale_y_continuous(labels = lambda l: ["{:,.0f}".format(v) for v in l]) +
            labs(x = "", y = "", title = title_l()) +
            theme_538() +
            theme(figure_size = (8, 5))
        )
    
    # Type of result ------------------------------------------------------------------------------------------------------------------------------
    if typ == "tbl": 
        return f_tbl 
    elif typ == "plt": 
        return f_plt
    else:
        raise ValueError(f"`typ` can either be 'tbl' or 'plt' and not {typ}")
        


# [] zone summary ==================================================================================================

def zone_summary(df, fun, cols = None):
    """
    parameters
    ----------
    df:   [DataFrame]
    fun:  [str] the summary function to use example :: 'mean', 'sum' or 'median'.
    cols: [list(float64, int64)] A list of variables from `df` to summarise.
    
    Return
    ------
    A summary table of all selected variables using the type of function `fun`.
    """
    # Conditions ------------------------------------------------------------------------------------------------------------------
    if cols is None:
        f_tbl_gp = df.groupby("Zone")[["PAYE", "Direct Assessment", "Road Tax", "Other Taxes", "Total Tax", "MDAs", "Grand Total"]]
    else:
        f_tbl_gp = df.groupby("Zone")[cols]
    
    if fun == "sum":
        f_tbl = f_tbl_gp.sum()
    elif fun == "mean":
        f_tbl = f_tbl_gp.mean()
    elif fun == "median":
        f_tbl = f_tbl_gp.median()
    else:
        raise ValueError(f"{fun} is not recognised use any of 'sum', 'mean' or median")
        
    return f_tbl.reset_index()



# [] source of zone IGR ============================================================================================
zone_color = {"South South": "#2E8B57",
              "South West": "#008000",
              "North Central" : "#008B00",
              "South East": "#238E23",
              "North West": "#32CD32",
              "North East": "#09F911"}

def zone_source(df, loc_var, period, fun = "sum", colr = None, typ = "plt"):
    """
    Parameter
    ---------
    df:      [dataframe]
    loc_var: [object] a geo-political zone location variable from the `df`.
    period:  [str] the period of the data e.i. ['q1', 'q2', 'half year'].
    fun:     [str] the type of summary function to use. ["mean", "sum", "median"].
    colr:    [list $ str] a list of 2 colors to use in the plot.
    typ:     [str] analysis to generate 'plt' for plot 'tbl' for table.
    
    returns
    -------
    if typ = 'plt' : plot
    if typ = 'tbl' : dataFrame
    """
    
    variables = ["PAYE", "Direct Assessment", "Road Tax", "Other Taxes", "MDAs"]
    f_tbl = zone_summary(df = df, fun = fun, cols = variables)
        
    f_tbl = f_tbl[f_tbl["Zone"] == loc_var].melt("Zone", variables, "revenue", "value")
    f_tbl["prop"] = prop_table(f_tbl, "value")
    
    # colors --------------------------------------------------------------------------------------
    colr = ["forestgreen", zone_color[loc_var]] if colr is None else colr
    
    # labels & title ------------------------------------------------------------------------------
    def title_l():
        period_dict = {"q1": "1st Quarter", "q2": "2nd Quater", "half year": "First Half"}
        try:
            period_value = period_dict[period]
            
            if period == "half year":
                if fun in ["mean", "median"]:
                    title_o = f"{period_value} Source Of {loc_var} Region Average IGR"
                else:
                    title_o = f"{period_value} Source Of {loc_var} Region Total IGR"

            else:
                if fun in ["mean", "median"]:
                    title_o = f"Source Of {loc_var} State Average IGR For The {period_value} Of 2021"
                else:
                    title_o = f"Source Of {loc_var} State Total IGR For The {period_value} Of 2021"
                    
            return title_o
        except:
            raise ValueError(f"{period} in not a valid period name")
            
    f_plt = (
            ggplot(f_tbl, aes("reorder(revenue, value)", "value")) +
            geom_segment(aes(xend = "revenue", y = 0,  yend = "value"), color = colr[0]) +
            geom_point(size = 16, color = colr[1]) +
            geom_text(aes(label = "prop"), position = position_dodge(0.9), color = "white", size = 8, fontweight = "bold", format_string = "{}%") +
            coord_flip() +
            scale_y_continuous(labels = lambda l: ["{:,.0f}".format(v) for v in l]) +
            labs(x = "", y = "", title = title_l()) +
            theme_538() +
            theme(figure_size = (8, 5))
        )
    
    if typ == "tbl":
        return f_tbl 
    elif typ == "plt": 
        return f_plt
    else:
        raise ValueError(f"`typ` can either be 'tbl' or 'plt' and not {typ}")




# [] Total revenue by Region =======================================================================================
def region_total_revenue(df, region, rev_typ, period):
    """
    Parameter
    ---------
    df:      [DataFrame]
    region:  [object, str] a region from the zone variable in `df`.
    rev_typ: [float64, int64] The type of revenue to plot.
    period:  [str] the period of the data e.i. ['q1', 'q2', 'half year'].
    
    Return
    ------
    a tree map of selected zones given the type of revenue `rev_typ`
    """
    # tbl ----------------------------------------------------------------------------------------------------------
    f_df = df[df["Zone"] == region]
    
    if type(rev_typ) != list:
        f_df = f_df[["State", rev_typ]]

        f_df["prop"]  = prop_table(f_df, rev_typ)
        f_df["label"]  = f_df["prop"].astype("str")+"%"
        f_df = f_df.rename(columns = {rev_typ: "valuez"})
        f_df["State"] = f_df["State"]+" State"
    else:
        f_df = f_df[["State"]+rev_typ]
        f_df["valuez"] = f_df[rev_typ].sum(1)
        f_df = f_df[["State", "valuez"]]

        f_df["prop"]  = prop_table(f_df, "valuez")
        f_df["label"]  = f_df["prop"].astype("str")+"%"
        f_df["State"] = f_df["State"]+" State"
        
    # Colors selection ----------------------------------------------------------------------------------------------
    state_count_dict = df.groupby("Zone")["State"].count().to_dict()
    num_state = state_count_dict[region]
    colr = ["firebrick", "forestgreen", "peru", "darkslategray", "navy", "magenta", "blueviolet"]
    plt_color = colr[:num_state]
    
    # plt label -----------------------------------------------------------------------------------------------------
    def title_l():
        period_dict = {"q1": "1st Quarter", "q2": "2nd Quater", "half year": "First Half"}
        period_value = period_dict[period]
            
        if type(rev_typ) != list:
            if period == "half year":
                title_o = f"Size Of {rev_typ} Revenue Generated From States In The {region} Region For The {period_value} Of 2021"
            else:
                title_o = f"Size Of {rev_typ} Revenue Generated From States In The {region} Region During The {period_value} Of 2021"

        else:
            tl = ", ".join(rev_typ)
            if period == "half year":
                title_o = f"Total Revenue Generated Through {tl} For States In The {region} Region For The {period_value} of 2021"
            else:
                title_o = f"Total Revenue Generated Through {tl} For States In The {region} Region During The {period_value} of 2021"
        return title_o
    
    # plt -----------------------------------------------------------------------------------------------------------
    plt.figure(figsize = (12, 9))
    sq.plot(sizes = f_df["valuez"].tolist(), 
        label = f_df["State"].tolist(), 
        value = f_df["label"].tolist(),
        color = plt_color,
        pad = True,
        text_kwargs = {"fontsize": 12, "fontweight": "bold", "color": "white"})
    
    plt.title(title_l(), fontdict = {"color": "White", "fontsize": 16})
    plt.axis("off")
    plt.show()



# [] percentage change in revenue ===================================================================================
def percent_change(df, old_value, new_value, deci = 2, chg_names = None, with_neg = True):
    """
    df: [DataFrame]
    old_value: [float64, int64] a variable from the `df` to use as the previous value.
    new_value: [float64, int64] a variable from the `df` to use as the new value.
    deci:      [int] the number of decimal point to keep when rounding.
    chg_names: [list(str)] Alternate names to give the two new columns.
    with_neg:  [bool] True to calculate with negative values, False to not.
    
    return
    ------
    two additional columns  if chg_names is none:
    difference: the difference between the new_value and the old_value
    percent_change: the increase or decrease change in percentage.
    """
    # Conditions -------------------------------------------------------------------------------------------
    if chg_names is None:
        if with_neg:
            df["difference"] = df[new_value] - df[old_value]
            df["percent_change"] = round(df["difference"] / df[old_value] * 100, deci)
        else:
            df["difference"] = abs(df[new_value] - df[old_value])
            df["percent_change"] = round(df["difference"] / df[old_value] * 100, deci)
    else:
        if len(chg_names) != 2:
            raise ValueError(f"`chg_names` must be a list of length 2, but {len(chg_names)} was given.")
            
        if with_neg:
            df[chg_names[0]] = df[new_value] - df[old_value]
            df[chg_names[1]] = round(df["difference"] / df[old_value] * 100, deci)
        else:
            df[chg_names[0]] = abs(df[new_value] - df[old_value])
            df[chg_names[1]] = round(df["difference"] / df[old_value] * 100, deci)
        
    return df


# [] Change in Revenue ===============================================================================================
def revenue_change(df1, df2, var, by = "State", smy_fun = "sum"):
    """
    Parameter
    ---------
    df1:     [DataFrame]
    df2:     [DataFrame]
    var:     [float64, int64] A variable to plot the percentage change.
    by:      [str] analysis by either 'state'- states or 'zone'- geo-political zone.
    smy_fun: [str] the type of Summary function to use when `by` == 'zone'. example: 'mean', 'sum', or 'median'
    
    Return
    ------
    A plot showing either a percentage increase or decrease in revenue.
    """
    # tbl -----------------------------------------------------------------------------------------------------------------
    f_df = df1[["State", "Zone", var]]
    
    def create_diff(f_dff):
        f_dff["new_value"] = df2[var]
        f_dff = percent_change(df = f_dff, old_value = var, new_value = "new_value", deci = 1)
        f_dff["pos_percent_chg"] = abs(f_dff.percent_change)
        f_dff["change"] = where(f_dff["percent_change"] >= 0, "Increase", "Decrease").tolist()
        f_dff = f_dff[~f_dff["percent_change"].isna()]
        return f_dff
    
    if by == "State":
        f_df = create_diff(f_df)
        
    elif by == "Zone":
        f_df = zone_summary(f_df, fun = smy_fun, cols = var)
        f_df = create_diff(f_df)
    else:
        raise ValueError(f"{by} is not recognised use any of 'state' or 'zone'")
        
    # Plt label ----------------------------------------------------------------------------------------------------------
    def title_l():
        if by == "State":
            title_o = f"Percentage Increase Or Decrease In {var} Revenue Generated By States In 1st-2nd Quarter Of 2021"
        elif by == "Zone":
            title_o = f"Percentage Increase Or Decrease In {var} Revenue Generated By Geo-Political Zones In 1st-2nd Quarter Of 2021"
            
        return title_o
    
    # plt ---------------------------------------------------------------------------------------------------------------
    f_plt = (
        ggplot(f_df, aes(x = by, y = "percent_change", fill = "change")) +
        geom_col(show_legend = "none") +
        coord_flip() +
        geom_text(aes(label = "pos_percent_chg", y = 0, ha = where(f_df["percent_change"] >= 0, "left", "right")), 
                  position = position_dodge(.9), size = 7, format_string = "{}%") +
        create_ordered_text(data = f_df, cat_col = by, num_col = "percent_change", sort_value = True, rev = True) +
        labs(x = "", y = "", title = title_l()) +
        theme_538() +
        theme(figure_size = (8, 6), plot_title = element_text(size = 10))
    )
    
    return f_plt



# [] overall summary of revenue from each source ===================================================================
def revenue_summary(df, period, func = "sum", colr = "seagreen"):
    """
    Parameters
    ---------
    df:     [DataFrame]
    period: [str] the period of the data e.i. ['q1', 'q2', 'half year'].
    func:   [str] the type of summary function to use. ["mean", "sum", "median"].
    colr:   [str] color of the plot bars.
    
    Return
    ------
    a plot with the percentage revenue each IGR generated in Total- 'sum' or on an Average- 'mean', 'median'
    """
    
    # tbl ---------------------------------------------------------------------------------------------------------
    drop_cols = ["State", "Zone", "Grand Total", "Total Tax"]
    f_df = df.drop(drop_cols, axis = 1)
    
    if func == "sum":
        f_df = f_df.sum()
    elif func == "mean":
        f_df = f_df.mean()
    elif func == "median":
        f_df = f_df.median()
    else:
        raise Exception(f"{fun} is not recognised use any of 'mean', 'sum' or 'median'")
        
    f_df = f_df.reset_index().rename(columns = {"index": "variable", 0: "value"})
    f_df["prop"] = prop_table(df = f_df, var = "value",  deci = 1)
    
    max_var_value = f_df["value"].max()
    text_pos = max_var_value - (0.05*max_var_value)
    
    # plt label ------------------------------------------------------------------------------------------------------
    def title_l():
        period_dict = {"q1": "1st Quarter", "q2": "2nd Quarter", "half year": "First Half"}
        period_value = period_dict[period]
        
        if period == "half year":
            if func == "sum":
                title_o = f"{period_value} Of 2021 Total Revenue Generated By Each Source Of IGR"
            else:
                title_o = f"{period_value} Of 2021 Average Revenue Generated By Each Source Of IGR"
        else:
            if func == "sum":
                title_o = f"Total Revenue Generated By Each Source Of IGR In The {period_value} Of 2021"
            else:
                title_o = f"Average Revenue Generated By Each Source Of IGR In The {period_value} Of 2021"
                
        return title_o
    
    # Plt ---------------------------------------------------------------------------------------------------------------
    f_plt = (
        ggplot(f_df, aes("reorder(variable, value)", "value")) +
        geom_col(fill = colr) +
        coord_flip() +
        geom_text(aes(label = "prop", y = text_pos), position = position_dodge(.9), format_string = "{}%") +
        scale_y_continuous(labels = lambda l: ["{:,.0f}".format(v) for v in l]) +
        labs(x = "Revenue", y = "", title = title_l()) +
        theme_538() +
        theme(figure_size= (9, 5))
    )
    
    return f_plt


