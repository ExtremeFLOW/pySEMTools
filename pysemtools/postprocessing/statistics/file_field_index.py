
def file_field_index(source_code: str = 'Neko', custom_file_names: dict = None):
    """
    Returns a dictionary with the file type and the field name for the statistics fields.
    """

    if source_code == 'Neko':
        if custom_file_names is not None:
            stats_field_name = custom_file_names.get('stats_field_name', 'fluid_stats0.fld')
        else:
            stats_field_name = 'fluid_stats0.fld'

        neko_stat_field_index = {}
        neko_stat_field_index['P'] = {'file_name' : stats_field_name, 'field_key' : 'pres', "registry" : 'p'}
        neko_stat_field_index['U'] = {'file_name' : stats_field_name, 'field_key' : 'vel_0', "registry" : 'u'}
        neko_stat_field_index['V'] = {'file_name' : stats_field_name, 'field_key' : 'vel_1', "registry" : 'v'}
        neko_stat_field_index['W'] = {'file_name' : stats_field_name, 'field_key' : 'vel_2', "registry" : 'w'}

        neko_stat_field_index['PP'] = {'file_name' : stats_field_name, 'field_key' : 'temp', "registry" : 't'}
        neko_stat_field_index['UU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_0', "registry" : 's0'}
        neko_stat_field_index['VV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_1', "registry" : 's1'}
        neko_stat_field_index['WW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_2', "registry" : 's2'}
        neko_stat_field_index['UV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_3', "registry" : 's3'}
        neko_stat_field_index['UW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_4', "registry" : 's4'}
        neko_stat_field_index['VW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_5', "registry" : 's5'}

        neko_stat_field_index['UUU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_6', "registry" : 's6'}
        neko_stat_field_index['VVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_7', "registry" : 's7'}
        neko_stat_field_index['WWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_8', "registry" : 's8'}
        neko_stat_field_index['UUV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_9', "registry" : 's9'}
        neko_stat_field_index['UUW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_10', "registry" : 's10'}
        neko_stat_field_index['UVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_11', "registry" : 's11'}
        neko_stat_field_index['UVW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_12', "registry" : 's12'}
        neko_stat_field_index['VVW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_13', "registry" : 's13'}
        neko_stat_field_index['UWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_14', "registry" : 's14'}
        neko_stat_field_index['VWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_15', "registry" : 's15'}

        neko_stat_field_index['UUUU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_16', "registry" : 's16'}
        neko_stat_field_index['VVVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_17', "registry" : 's17'}
        neko_stat_field_index['WWWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_18', "registry" : 's18'}
        neko_stat_field_index['PPP'] = {'file_name' : stats_field_name, 'field_key' : 'scal_19', "registry" : 's19'}
        neko_stat_field_index['PPPP'] = {'file_name' : stats_field_name, 'field_key' : 'scal_20', "registry" : 's20'}
        neko_stat_field_index['PU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_21', "registry" : 's21'}
        neko_stat_field_index['PV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_22', "registry" : 's22'}
        neko_stat_field_index['PW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_23', "registry" : 's23'}

        neko_stat_field_index['PDU/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_24', "registry" : 's24'}
        neko_stat_field_index['PDU/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_25', "registry" : 's25'}
        neko_stat_field_index['PDU/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_26', "registry" : 's26'}
        neko_stat_field_index['PDV/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_27', "registry" : 's27'}
        neko_stat_field_index['PDV/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_28', "registry" : 's28'}
        neko_stat_field_index['PDV/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_29', "registry" : 's29'}
        neko_stat_field_index['PDW/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_30', "registry" : 's30'}
        neko_stat_field_index['PDW/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_31', "registry" : 's31'}
        neko_stat_field_index['PDW/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_32', "registry" : 's32'}

        neko_stat_field_index['E11'] = {'file_name' : stats_field_name, 'field_key' : 'scal_33', "registry" : 's33'}
        neko_stat_field_index['E22'] = {'file_name' : stats_field_name, 'field_key' : 'scal_34', "registry" : 's34'}
        neko_stat_field_index['E33'] = {'file_name' : stats_field_name, 'field_key' : 'scal_35', "registry" : 's35'}
        neko_stat_field_index['E12'] = {'file_name' : stats_field_name, 'field_key' : 'scal_36', "registry" : 's36'}
        neko_stat_field_index['E13'] = {'file_name' : stats_field_name, 'field_key' : 'scal_37', "registry" : 's37'}
        neko_stat_field_index['E23'] = {'file_name' : stats_field_name, 'field_key' : 'scal_38', "registry" : 's38'}

        neko_stat_field_index['W_2D'] = {'file_name' : stats_field_name, 'field_key' : 'scal_39', "registry" : 's39'}
        stat_field_index = neko_stat_field_index

    elif source_code == 'Neko_old':

        if not isinstance(custom_file_names, type(None)):
            mean_field_name = custom_file_names.get('mean_field_name', 'mean_field.fld')
            stats_field_name = custom_file_names.get('stats_field_name', 'stats.fld')
        else:
            mean_field_name = 'mean_field.fld'
            stats_field_name = 'stats.fld'

        neko_stat_field_index = {}

        neko_stat_field_index['P'] = {'file_name' : mean_field_name, 'field_key' : 'pres', "pynek_field_key" : 'p'}
        neko_stat_field_index['U'] = {'file_name' : mean_field_name, 'field_key' : 'vel_0', "pynek_field_key" : 'u'}
        neko_stat_field_index['V'] = {'file_name' : mean_field_name, 'field_key' : 'vel_1', "pynek_field_key" : 'v'}
        neko_stat_field_index['W'] = {'file_name' : mean_field_name, 'field_key' : 'vel_2', "pynek_field_key" : 'w'}

        neko_stat_field_index['PP'] = {'file_name' : stats_field_name, 'field_key' : 'pres', "pynek_field_key" : 'p'}
        neko_stat_field_index['UU'] = {'file_name' : stats_field_name, 'field_key' : 'vel_0', "pynek_field_key" : 'u'}
        neko_stat_field_index['VV'] = {'file_name' : stats_field_name, 'field_key' : 'vel_1', "pynek_field_key" : 'v'}
        neko_stat_field_index['WW'] = {'file_name' : stats_field_name, 'field_key' : 'vel_2', "pynek_field_key" : 'w'}
        
        neko_stat_field_index['UV'] = {'file_name' : stats_field_name, 'field_key' : 'temp', "pynek_field_key" : 't'}
        neko_stat_field_index['UW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_0', "pynek_field_key" : 's0'}
        neko_stat_field_index['VW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_1', "pynek_field_key" : 's1'}

        neko_stat_field_index['UUU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_2', "pynek_field_key" : 's2'}
        neko_stat_field_index['VVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_3', "pynek_field_key" : 's3'}
        neko_stat_field_index['WWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_4', "pynek_field_key" : 's4'}

        neko_stat_field_index['UUV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_5', "pynek_field_key" : 's5'}
        neko_stat_field_index['UUW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_6', "pynek_field_key" : 's6'}
        neko_stat_field_index['UVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_7', "pynek_field_key" : 's7'}
        neko_stat_field_index['UVW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_8', "pynek_field_key" : 's8'}
        neko_stat_field_index['VVW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_9', "pynek_field_key" : 's9'}
        neko_stat_field_index['UWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_10', "pynek_field_key" : 's10'}
        neko_stat_field_index['VWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_11', "pynek_field_key" : 's11'}
        
        neko_stat_field_index['UUUU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_12', "pynek_field_key" : 's12'}
        neko_stat_field_index['VVVV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_13', "pynek_field_key" : 's13'}
        neko_stat_field_index['WWWW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_14', "pynek_field_key" : 's14'}
        
        neko_stat_field_index['PPP'] = {'file_name' : stats_field_name, 'field_key' : 'scal_15', "pynek_field_key" : 's15'}
        neko_stat_field_index['PPPP'] = {'file_name' : stats_field_name, 'field_key' : 'scal_16', "pynek_field_key" : 's16'}

        neko_stat_field_index['PU'] = {'file_name' : stats_field_name, 'field_key' : 'scal_17', "pynek_field_key" : 's17'}
        neko_stat_field_index['PV'] = {'file_name' : stats_field_name, 'field_key' : 'scal_18', "pynek_field_key" : 's18'}
        neko_stat_field_index['PW'] = {'file_name' : stats_field_name, 'field_key' : 'scal_19', "pynek_field_key" : 's19'}

        neko_stat_field_index['PDU/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_20', "pynek_field_key" : 's20'}
        neko_stat_field_index['PDU/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_21', "pynek_field_key" : 's21'}
        neko_stat_field_index['PDU/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_22', "pynek_field_key" : 's22'}
        
        neko_stat_field_index['PDV/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_23', "pynek_field_key" : 's23'}
        neko_stat_field_index['PDV/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_24', "pynek_field_key" : 's24'}
        neko_stat_field_index['PDV/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_25', "pynek_field_key" : 's25'}
        
        neko_stat_field_index['PDW/DX'] = {'file_name' : stats_field_name, 'field_key' : 'scal_26', "pynek_field_key" : 's26'}
        neko_stat_field_index['PDW/DY'] = {'file_name' : stats_field_name, 'field_key' : 'scal_27', "pynek_field_key" : 's27'}
        neko_stat_field_index['PDW/DZ'] = {'file_name' : stats_field_name, 'field_key' : 'scal_28', "pynek_field_key" : 's28'}

        neko_stat_field_index['E11'] = {'file_name' : stats_field_name, 'field_key' : 'scal_29', "pynek_field_key" : 's29'}
        neko_stat_field_index['E22'] = {'file_name' : stats_field_name, 'field_key' : 'scal_30', "pynek_field_key" : 's30'}
        neko_stat_field_index['E33'] = {'file_name' : stats_field_name, 'field_key' : 'scal_31', "pynek_field_key" : 's31'}
        neko_stat_field_index['E12'] = {'file_name' : stats_field_name, 'field_key' : 'scal_32', "pynek_field_key" : 's32'}
        neko_stat_field_index['E13'] = {'file_name' : stats_field_name, 'field_key' : 'scal_33', "pynek_field_key" : 's33'}
        neko_stat_field_index['E23'] = {'file_name' : stats_field_name, 'field_key' : 'scal_34', "pynek_field_key" : 's34'}


        stat_field_index = neko_stat_field_index
    
    else:
        raise ValueError('The source code is not supported. Please use "Neko" as the source code.')

    return stat_field_index
