## Wind Data Format

The pickle files contained in this directory contain wind data over multiple
years. Refer to wind_data_buoy.ipynb for reference on how to work with this
data.

Example code for retrieving time and windspeed data from the opened pickle file:

    path = "pickle/dataOffshore.pickle"
    data_offshore = pickle.load( open( path, "rb" ) )

    # convert seconds to datetime
    time_offshore = data_offshore.time
    time_offshore = time_offshore.apply(lib.ntp_seconds_to_datetime)

    # get wind data
    wind_mag_offshore = np.sqrt((data_offshore.eastward_wind_velocity)**2 + (data_offshore.northward_wind_velocity)**2)
    wind_angle_offshore = np.arctan2(data_offshore.northward_wind_velocity, data_offshore.eastward_wind_velocity)

Data downloaded on 4/24/2021 with these settings for Oregon Offshore:

    ## get URLs for mooring data
    urls_offshore = lib.web_crawler_mooring('2015-05-01T00:00:00.000Z', '2020-02-20T12:00:00.000Z', location='offshore')

    # get mooring data
    print(urls_offshore)
    data_offshore = lib.get_mooring_data(urls_offshore)

Data downloaded on 4/24/2021 with these settings for Oregon Shelf:

    ## get URLs for mooring data
    urls_shelf = lib.web_crawler_mooring('2015-05-01T00:00:00.000Z', '2020-02-20T12:00:00.000Z', location='shelf')

    # get mooring data
    print(urls_shelf)
    data_shelf = lib.get_mooring_data(urls_shelf)
