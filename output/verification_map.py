from data_import.input_output import load_csv
import plotly.graph_objects as go
import pandas as pd
# dash imports unused: the only Dash usage in this file is commented out (see bottom). Re-enable if reviving the dashboard.
# import dash
# from dash import dcc
# from dash import html
# import dash_core_components as dcc # TODO old version of Dash (changed in revision 14 opSim environment)
# import dash_html_components as html
from config import directories, RUN_CONFIG

TRANSITION_DURATION = 10
FRAME_DURATION = 10
AIRCRAFT_COLOR = {'flying':'purple',
                  'on ground':'green'
                  }

def simulation_map():
    log_name = RUN_CONFIG.MAP_LOG
    # Load airports data
    df_airports = load_csv('Airports')
    df_airports_coordinates = load_csv('AirportsCoordinates')
    df_airports_coordinates = df_airports_coordinates.drop(labels=['AirportName'], axis=1)
    df_airports.drop_duplicates(subset=['IcaoAirportCode', 'IataAirportCode'], keep='first', inplace=True)
    df_airports.sort_values(by=['IataAirportCode'], ascending=True, inplace=True)
    df_airports = pd.merge(df_airports, df_airports_coordinates, left_on='IcaoAirportCode', right_on='AirportICAO', how='left')

    # Load log
    df_log = load_csv(log_name, directory_input=directories.logs_map)


    ####################### LAYOUT #######################
    # Define buttons
    button_play = {
        "args": [None, {"frame": {"duration": FRAME_DURATION,"redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": TRANSITION_DURATION,
                                       "easing": "quadratic-in-out"}}],
        "label": "Play",
        "method": "animate"
    }
    button_pause = {
        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                          "mode": "immediate",
                          "transition": {"duration": 0}}],
        "label": "Pause",
        "method": "animate"
    }

    buttons = [
        {   'buttons': [button_play, button_pause],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
            }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Simulation time (UTC):",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # Figure layout
    layout = dict(
        title_text='Operations simulation',
        geo=go.layout.Geo(scope='world',
                          projection=dict(type='natural earth'),
                          showland=True,
                          landcolor='rgb(243,243,243)',
                          countrycolor='rgb(204,204,204)',
                          ),
        showlegend=False,
        updatemenus=buttons
    )

    ####################### AIRPORTS AND ROUTES #######################

    # Find list of latitudes and longitudes
    drawn_airports = []
    airport_latitudes = []
    airport_longitudes = []
    airport_text = []

    drawn_routes = []
    routes_latitudes = []
    routes_longitudes = []

    def nan_to_empty_string(text):
        ''' Check if input is string and if it is not makes it into an empty string'''
        if isinstance(text, str) == 0:
            text = ''
        return text

    def add_airport(row):
        ''' Given a row referring to an airports, add the relevant data to lists for drawing airports on map'''
        # Latitude and longitude
        airport_latitudes.append(row['Latitude'])
        airport_longitudes.append(row['Longitude'])

        # Hovertext
        airport_name = row['AirportName']
        airport_name = nan_to_empty_string(airport_name)
        IATA = row['IataAirportCode']
        IATA = nan_to_empty_string(IATA)
        ICAO = row['IcaoAirportCode']
        ICAO = nan_to_empty_string(ICAO)
        hovertext = airport_name+'<br>' \
                    +'IATA: '+IATA +'<br>' \
                    +'ICAO: '+ICAO
        airport_text.append(hovertext)


    # Filter dataframes to inclue only flights and drop origin-destination duplicates
    df_only_flights = df_log[df_log['State']=='flying']
    df_only_flights = df_only_flights.drop_duplicates(['FlightOrig','FlightDest'])
    for index, route_row in df_only_flights.iterrows():
        # Find names for route
        ap_dep = route_row['FlightOrig']
        ap_arr = route_row['FlightDest']
        # Skip route if it has already been drawn
        if (ap_dep, ap_arr) in drawn_routes or (ap_arr, ap_dep) in drawn_routes:
            continue
        # Else add coordinates to routes to be drawn
        row_departure = df_airports[df_airports['IataAirportCode']==ap_dep].iloc[0]
        row_arrival =  df_airports[df_airports['IataAirportCode']==ap_arr].iloc[0]
        # Append route latitude data
        routes_latitudes.append(row_departure['Latitude'])
        routes_latitudes.append(row_arrival['Latitude'])
        routes_latitudes.append(None)
        # Append route longitude data
        routes_longitudes.append(row_departure['Longitude'])
        routes_longitudes.append(row_arrival['Longitude'])
        routes_longitudes.append(None)
        drawn_routes.append((ap_dep, ap_arr))


        # If airport not drawn yet, draw it
        if ap_dep not in drawn_airports:
            add_airport(row_departure)
            drawn_airports.append(ap_dep)
        if ap_arr not in drawn_airports:
            add_airport(row_arrival)
            drawn_airports.append(ap_arr)

    plot_airports = go.Scattergeo(
        name='Airports',
        lon=airport_longitudes,
        lat=airport_latitudes,
        hoverinfo='text',
        text=airport_text,
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            opacity=0.5,
            line=dict(
                width=10,
                color='rgba(68,68,68,0)'
            )
        )
    )

    plot_routes = go.Scattergeo(
        name='Routes',
        lon=routes_longitudes,
        lat=routes_latitudes,
        hoverinfo='none',
        mode='lines',
        line=dict(width=1, color='blue'),
        opacity=0.5
    )

    # Note: Airports must be passed as input twice, otherwise they will be cancelled after the first frame
    data_fixed = [plot_airports, plot_routes, plot_airports]

    ####################### FRAMES #######################
    frames = []
    # Find list of logged times
    times = list(set(df_log['TimeSimulation'].to_list()))
    times = sorted(times)

    # Add columns for hover info in dataframe
    df_log['FlightShortId'] = df_log['Flight'].str[11:18]
    df_log['FlightShortId'] = df_log['FlightShortId'].fillna('')
    df_log['HoverInfo'] = df_log['AircraftId'] + '<br>' \
                          +'Subtype: ' + df_log['Subtype'] +'<br>' \
                          +'Flight: ' + df_log['FlightShortId']

    # Generate a frame for each time
    for time in times:
        # Filter the log dataframe
        df_log_time = df_log[df_log['TimeSimulation']==time]
        # Generate frame dict
        frame_dict = {'data':[],
                      'name': str(time)
                      }

        # Generate a graphical object for the aircraft
        plot_aircraft = go.Scattergeo(
            lon=df_log_time['Longitude'],
            lat=df_log_time['Latitude'],
            ids=df_log_time['AircraftId'].to_list(),
            name='Aircraft',
            uid='Aircraft',
            hovertext=df_log_time['HoverInfo'],
            text=df_log_time['AircraftId'],
            textposition='top center',
            mode='markers+text',
            marker=dict(
                size=15,
                color=[AIRCRAFT_COLOR[st] for st in df_log_time['State'].to_list()],
                line=dict(
                    width=10,
                    color='rgba(68,68,68,0)'
                )
            )
        )
        frame_dict['data'].append(plot_aircraft)

        # Generate frame from dictionary
        frame_new = go.Frame(frame_dict)
        frames.append(frame_new)

        # Update slider steps
        slider_step = {'args': [
            [time],
            {'frame': {"duration": FRAME_DURATION, "redraw": True},
             "mode": "immediate",
             "transition": {"duration": TRANSITION_DURATION}}
        ],
            'label':df_log_time['Time'].iloc[0],
            'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
        layout['sliders'] = [sliders_dict]

    ####################### DRAW FIGURE #######################

    fig = go.Figure({
        'data': data_fixed,
        'layout': layout,
        'frames': frames})

    fig.show()

    # CODE TO SHOW IN DASH
    # app = dash.Dash()
    # app.layout = html.Div([
    #     dcc.Graph(figure=fig)
    # ])
    # app.run_server(debug=True, use_reloader=False)

    print('done')


