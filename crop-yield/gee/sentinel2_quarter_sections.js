/*
  Sentinel-2 quarter-section feature export for Alberta.

  Purpose:
  - Read uploaded Alberta quarter-section polygons from a GEE table asset
  - Build seasonal Sentinel-2 summary features for each quarter section and year
  - Export a flat table for local machine learning

  Before running:
  1. Upload your Alberta quarter-section GeoJSON to GEE as a table asset
  2. Set QUARTER_SECTIONS_ASSET below
  3. Set QUARTER_ID_FIELD to a stable ID column from your uploaded asset
  4. Optionally filter the collection to a smaller pilot region first
*/

var QUARTER_SECTIONS_ASSET = 'users/your_username/alberta_quarter_sections';
var QUARTER_ID_FIELD = 'PID';
var START_YEAR = 2018;
var END_YEAR = 2023;
var EXPORT_DESCRIPTION = 'alberta_quarter_sections_sentinel2_features';
var EXPORT_FOLDER = 'gee_exports';
var EXPORT_FILE_PREFIX = 'alberta_quarter_sections_s2_features';

// Optional: reduce the run size while testing.
var LIMIT_FEATURES = null; // e.g. 500
var COVERAGE_MIN_VALID_PIXELS = 50;
var COVERAGE_REQUIRED_WINDOWS = 2;
var COVERAGE_YEARS = [2021, 2022, 2023];
var COVERAGE_REQUIRED_YEARS = 2;

var quarterSections = ee.FeatureCollection(QUARTER_SECTIONS_ASSET);
if (LIMIT_FEATURES !== null) {
  quarterSections = quarterSections.limit(LIMIT_FEATURES);
}

Map.centerObject(quarterSections, 6);
Map.addLayer(quarterSections, {}, 'quarter sections');

var albers = 'EPSG:3347';
var AAFC_ACI = ee.ImageCollection('AAFC/ACI');
var SMAP = ee.ImageCollection('NASA/SMAP/SPL3SMP_E/005')
  .merge(ee.ImageCollection('NASA/SMAP/SPL3SMP_E/006'));
var ERA5_LAND = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
var DEM = ee.Image('USGS/SRTMGL1_003');
var SOIL_ORGANIC_CARBON = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02');
var SOIL_SAND = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02');
var SOIL_PH = ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02');

var S2_FEATURE_BANDS = [
  'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12',
  'NDVI', 'EVI', 'NDRE1', 'NDRE2', 'GNDVI', 'NDMI', 'MSI', 'OSAVI', 'NBR2', 'RECI'
];

var S2_REDUCER_SUFFIXES = ['mean', 'stdDev', 'p10', 'p50', 'p90'];

var SMAP_FEATURE_BANDS = [
  'soil_moisture_am',
  'soil_moisture_pm',
  'vegetation_water_content_am',
  'vegetation_water_content_pm'
];

var SMAP_REDUCER_SUFFIXES = ['mean', 'stdDev', 'p10', 'p50', 'p90'];
var WEATHER_FEATURE_BANDS = [
  'temperature_2m',
  'temperature_2m_max',
  'total_precipitation_sum',
  'dewpoint_temperature_2m'
];
var WEATHER_REDUCER_SUFFIXES = ['mean', 'sum', 'p10', 'p50', 'p90'];
var STATIC_FEATURE_BANDS = [
  'elevation',
  'slope',
  'soil_organic_carbon_0cm',
  'soil_sand_0cm',
  'soil_ph_h2o_0cm'
];
var STATIC_REDUCER_SUFFIXES = ['mean', 'stdDev', 'p10', 'p50', 'p90'];

function addSpectralIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI');
  var ndre1 = image.normalizedDifference(['B8', 'B5']).rename('NDRE1');
  var ndre2 = image.normalizedDifference(['B8', 'B6']).rename('NDRE2');
  var gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI');
  var msi = image.select('B11').divide(image.select('B8')).rename('MSI');
  var reci = image.select('B8').divide(image.select('B5')).subtract(1).rename('RECI');
  var osavi = image.expression(
    '1.16 * ((nir - red) / (nir + red + 0.16))',
    {
      nir: image.select('B8'),
      red: image.select('B4')
    }
  ).rename('OSAVI');
  var nbr2 = image.normalizedDifference(['B11', 'B12']).rename('NBR2');
  var evi = image.expression(
    '2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))',
    {
      nir: image.select('B8'),
      red: image.select('B4'),
      blue: image.select('B2')
    }
  ).rename('EVI');

  return image.addBands([
    ndvi, evi, ndre1, ndre2, gndvi, ndmi, msi, osavi, nbr2, reci
  ]);
}

function maskS2Clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var clear = qa.bitwiseAnd(cloudBitMask).eq(0)
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(clear).copyProperties(image, ['system:time_start']);
}

function getS2Collection(startDate, endDate, region) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 50))
    .map(maskS2Clouds)
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
    .map(addSpectralIndices);
}

function buildReducer() {
  return ee.Reducer.mean()
    .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.percentile([10, 50, 90]), sharedInputs: true});
}

function buildWeatherReducer() {
  return ee.Reducer.mean()
    .combine({reducer2: ee.Reducer.sum(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.percentile([10, 50, 90]), sharedInputs: true});
}

function getSmapCollection(startDate, endDate, region) {
  return SMAP
    .filterBounds(region)
    .filterDate(startDate, endDate)
    .select(SMAP_FEATURE_BANDS);
}

function getWeatherCollection(startDate, endDate, region) {
  return ERA5_LAND
    .filterBounds(region)
    .filterDate(startDate, endDate)
    .select(WEATHER_FEATURE_BANDS);
}

function summarizeAafcCropType(feature, year) {
  var geom = feature.geometry();
  var startDate = ee.Date.fromYMD(year, 1, 1);
  var endDate = startDate.advance(1, 'year');
  var aafcImage = AAFC_ACI
    .filterDate(startDate, endDate)
    .first();

  var empty = ee.Dictionary({
    aafc_dominant_class: null,
    aafc_canola_fraction: null,
    aafc_wheat_fraction: null,
    aafc_spring_wheat_fraction: null,
    aafc_barley_fraction: null,
    aafc_oats_fraction: null,
    aafc_peas_fraction: null,
    aafc_lentils_fraction: null,
    aafc_ag_fraction: null
  });

  return ee.Dictionary(
    ee.Algorithms.If(
      aafcImage,
      ee.Dictionary({
        aafc_dominant_class: ee.Image(aafcImage).select('landcover').reduceRegion({
          reducer: ee.Reducer.mode(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e9,
          tileScale: 4
        }).get('landcover'),
        aafc_canola_fraction: ee.Image(aafcImage).select('landcover').eq(153).reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e9,
          tileScale: 4
        }).get('landcover'),
        aafc_wheat_fraction: ee.Image(aafcImage).select('landcover').eq(140).reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e9,
          tileScale: 4
        }).get('landcover'),
        aafc_spring_wheat_fraction: ee.Image(aafcImage).select('landcover').eq(146).reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e9,
          tileScale: 4
        }).get('landcover'),
        aafc_barley_fraction: ee.Image(aafcImage).select('landcover').eq(133).reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e9,
          tileScale: 4
        }).get('landcover'),
        aafc_oats_fraction: ee.Image(aafcImage).select('landcover').eq(136).reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e9,
          tileScale: 4
        }).get('landcover'),
        aafc_peas_fraction: ee.Image(aafcImage).select('landcover').eq(162).reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e9,
          tileScale: 4
        }).get('landcover'),
        aafc_lentils_fraction: ee.Image(aafcImage).select('landcover').eq(174).reduceRegion({
          reducer: ee.Reducer.mean(),
          geometry: geom,
          scale: 30,
          maxPixels: 1e9,
          tileScale: 4
        }).get('landcover'),
        aafc_ag_fraction: ee.Image(aafcImage).select('landcover').gte(120)
          .and(ee.Image(aafcImage).select('landcover').lt(200))
          .reduceRegion({
            reducer: ee.Reducer.mean(),
            geometry: geom,
            scale: 30,
            maxPixels: 1e9,
            tileScale: 4
          }).get('landcover')
      }),
      empty
    )
  );
}

var seasonalWindows = [
  {name: 'may_jun', startMonth: 5, startDay: 1, endMonth: 6, endDay: 15},
  {name: 'late_jun_jul', startMonth: 6, startDay: 16, endMonth: 7, endDay: 31},
  {name: 'aug', startMonth: 8, startDay: 1, endMonth: 8, endDay: 31},
  {name: 'sep', startMonth: 9, startDay: 1, endMonth: 9, endDay: 30}
];

function getWindowDates(year, windowDef) {
  windowDef = ee.Dictionary(windowDef);
  return {
    startDate: ee.Date.fromYMD(
      year,
      ee.Number(windowDef.get('startMonth')),
      ee.Number(windowDef.get('startDay'))
    ),
    endDate: ee.Date.fromYMD(
      year,
      ee.Number(windowDef.get('endMonth')),
      ee.Number(windowDef.get('endDay'))
    ).advance(1, 'day')
  };
}

function getValidPixelCount(feature, year, windowDef) {
  var geom = feature.geometry();
  var dates = getWindowDates(year, windowDef);
  var collection = getS2Collection(dates.startDate, dates.endDate, geom);

  var count = ee.Algorithms.If(
    collection.size().gt(0),
    collection.select('B8').median().mask().reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: geom,
      scale: 10,
      maxPixels: 1e9,
      tileScale: 4
    }).get('B8'),
    0
  );

  return ee.Number(ee.Algorithms.If(count, count, 0));
}

function addCoverageDiagnostics(feature) {
  var diagnostics = ee.Dictionary(
    ee.List(COVERAGE_YEARS).iterate(function(yearValue, acc) {
      var year = ee.Number(yearValue).toInt();
      acc = ee.Dictionary(acc);

      var yearWindowCounts = ee.Dictionary(
        ee.List(seasonalWindows).iterate(function(windowDef, innerAcc) {
          windowDef = ee.Dictionary(windowDef);
          innerAcc = ee.Dictionary(innerAcc);
          var windowName = ee.String(windowDef.get('name'));
          var validPixels = getValidPixelCount(feature, year, windowDef);
          return innerAcc.set(
            ee.String('coverage_')
              .cat(year.format())
              .cat('_')
              .cat(windowName)
              .cat('_valid_px'),
            validPixels
          );
        }, ee.Dictionary({}))
      );

      var validWindowCount = ee.Number(
        ee.List(seasonalWindows).iterate(function(windowDef, n) {
          windowDef = ee.Dictionary(windowDef);
          n = ee.Number(n);
          var validPixels = getValidPixelCount(feature, year, windowDef);
          return ee.Algorithms.If(
            validPixels.gte(COVERAGE_MIN_VALID_PIXELS),
            n.add(1),
            n
          );
        }, 0)
      );

      return acc
        .combine(yearWindowCounts, true)
        .set(
          ee.String('coverage_')
            .cat(year.format())
            .cat('_valid_window_count'),
          validWindowCount
        );
    }, ee.Dictionary({}))
  );

  var eligibleYears = ee.Number(
    ee.List(COVERAGE_YEARS).iterate(function(yearValue, n) {
      var year = ee.Number(yearValue).toInt();
      n = ee.Number(n);
      var yearValidWindows = ee.Number(
        diagnostics.get(
          ee.String('coverage_').cat(year.format()).cat('_valid_window_count')
        )
      );
      return ee.Algorithms.If(
        yearValidWindows.gte(COVERAGE_REQUIRED_WINDOWS),
        n.add(1),
        n
      );
    }, 0)
  );

  return feature
    .set(diagnostics)
    .set('coverage_eligible_year_count', eligibleYears);
}

function buildEmptyWindowSummary(windowName, featureBands, reducerSuffixes) {
  windowName = ee.String(windowName);
  var keys = ee.List(featureBands).map(function(band) {
    band = ee.String(band);
    return ee.List(reducerSuffixes).map(function(suffix) {
      suffix = ee.String(suffix);
      return windowName.cat('_').cat(band).cat('_').cat(suffix);
    });
  }).flatten();

  var values = keys.map(function(_) {
    return -9999;
  });

  return ee.Dictionary.fromLists(keys, values)
    .set(windowName.cat('_obs_count'), 0);
}

function summarizeWeatherYear(feature, year) {
  var geom = feature.geometry();
  var startDate = ee.Date.fromYMD(year, 4, 1);
  var endDate = ee.Date.fromYMD(year, 9, 30).advance(1, 'day');
  var weatherName = ee.String('grow_season_weather');
  var collection = getWeatherCollection(startDate, endDate, geom);
  var imageCount = collection.size();
  var emptySummary = buildEmptyWindowSummary(weatherName, WEATHER_FEATURE_BANDS, WEATHER_REDUCER_SUFFIXES);

  var composite = collection.mean();
  var reduced = ee.Algorithms.If(
    imageCount.gt(0),
    composite.reduceRegion({
      reducer: buildWeatherReducer(),
      geometry: geom,
      scale: 11132,
      maxPixels: 1e9,
      tileScale: 4
    }),
    ee.Dictionary({})
  );

  reduced = ee.Dictionary(reduced).map(function(key, value) {
    return value;
  });

  var renamed = reduced.keys().iterate(function(key, acc) {
    key = ee.String(key);
    acc = ee.Dictionary(acc);
    return acc.set(weatherName.cat('_').cat(key), reduced.get(key));
  }, emptySummary);

  return ee.Dictionary(renamed).set(weatherName.cat('_obs_count'), imageCount);
}

function summarizeStaticContext(feature) {
  var geom = feature.geometry();
  var slope = ee.Terrain.slope(DEM).rename('slope');
  var staticImage = DEM.rename('elevation')
    .addBands(slope)
    .addBands(SOIL_ORGANIC_CARBON.select('b0').multiply(5).rename('soil_organic_carbon_0cm'))
    .addBands(SOIL_SAND.select('b0').rename('soil_sand_0cm'))
    .addBands(SOIL_PH.select('b0').multiply(0.1).rename('soil_ph_h2o_0cm'));

  var emptySummary = buildEmptyWindowSummary('static', STATIC_FEATURE_BANDS, STATIC_REDUCER_SUFFIXES);
  var reduced = staticImage.reduceRegion({
    reducer: buildReducer(),
    geometry: geom,
    scale: 30,
    maxPixels: 1e9,
    tileScale: 4
  });

  reduced = ee.Dictionary(reduced).map(function(key, value) {
    return value;
  });

  var renamed = reduced.keys().iterate(function(key, acc) {
    key = ee.String(key);
    acc = ee.Dictionary(acc);
    return acc.set(ee.String('static_').cat(key), reduced.get(key));
  }, emptySummary);

  return ee.Dictionary(renamed);
}

function summarizeWindow(feature, year, windowDef) {
  windowDef = ee.Dictionary(windowDef);
  var geom = feature.geometry();
  var windowName = ee.String(windowDef.get('name'));
  var dates = getWindowDates(year, windowDef);
  var startDate = dates.startDate;
  var endDate = dates.endDate;

  var collection = getS2Collection(startDate, endDate, geom);
  var imageCount = collection.size();
  var emptySummary = buildEmptyWindowSummary(windowName, S2_FEATURE_BANDS, S2_REDUCER_SUFFIXES);

  var composite = collection.select(S2_FEATURE_BANDS).median();
  var reduced = ee.Algorithms.If(
    imageCount.gt(0),
    composite.reduceRegion({
      reducer: buildReducer(),
      geometry: geom,
      scale: 10,
      maxPixels: 1e9,
      tileScale: 4
    }),
    ee.Dictionary({})
  );

  reduced = ee.Dictionary(reduced).map(function(key, value) {
    return value;
  });

  var renamed = reduced.keys().iterate(function(key, acc) {
    key = ee.String(key);
    acc = ee.Dictionary(acc);
    return acc.set(windowName.cat('_').cat(key), reduced.get(key));
  }, emptySummary);

  return ee.Dictionary(renamed).set(windowName.cat('_obs_count'), imageCount);
}

function summarizeSmapWindow(feature, year, windowDef) {
  windowDef = ee.Dictionary(windowDef);
  var geom = feature.geometry();
  var windowName = ee.String(windowDef.get('name')).cat('_smap');
  var dates = getWindowDates(year, windowDef);
  var startDate = dates.startDate;
  var endDate = dates.endDate;

  var collection = getSmapCollection(startDate, endDate, geom);
  var imageCount = collection.size();
  var emptySummary = buildEmptyWindowSummary(windowName, SMAP_FEATURE_BANDS, SMAP_REDUCER_SUFFIXES);

  var composite = collection.median();
  var reduced = ee.Algorithms.If(
    imageCount.gt(0),
    composite.reduceRegion({
      reducer: buildReducer(),
      geometry: geom,
      scale: 9000,
      maxPixels: 1e9,
      tileScale: 4
    }),
    ee.Dictionary({})
  );

  reduced = ee.Dictionary(reduced).map(function(key, value) {
    return value;
  });

  var renamed = reduced.keys().iterate(function(key, acc) {
    key = ee.String(key);
    acc = ee.Dictionary(acc);
    return acc.set(windowName.cat('_').cat(key), reduced.get(key));
  }, emptySummary);

  return ee.Dictionary(renamed).set(windowName.cat('_obs_count'), imageCount);
}

function buildYearFeature(feature, year) {
  year = ee.Number(year).toInt();
  var base = ee.Dictionary({
    quarter_id: feature.get(QUARTER_ID_FIELD),
    year: year
  });

  var withMeta = base
    .set('M', feature.get('M'))
    .set('RGE', feature.get('RGE'))
    .set('TWP', feature.get('TWP'))
    .set('SEC', feature.get('SEC'))
    .set('QS', feature.get('QS'))
    .set('RA', feature.get('RA'))
    .set('coverage_eligible_year_count', feature.get('coverage_eligible_year_count'));

  var seasonal = ee.Dictionary(
    ee.List(seasonalWindows).iterate(function(windowDef, acc) {
      windowDef = ee.Dictionary(windowDef);
      acc = ee.Dictionary(acc);
      var summary = summarizeWindow(feature, year, windowDef);
      return acc.combine(summary, true);
    }, ee.Dictionary({}))
  );

  var seasonalSmap = ee.Dictionary(
    ee.List(seasonalWindows).iterate(function(windowDef, acc) {
      windowDef = ee.Dictionary(windowDef);
      acc = ee.Dictionary(acc);
      var summary = summarizeSmapWindow(feature, year, windowDef);
      return acc.combine(summary, true);
    }, ee.Dictionary({}))
  );

  var weatherSummary = summarizeWeatherYear(feature, year);
  var staticSummary = summarizeStaticContext(feature);
  var aafcCrop = summarizeAafcCropType(feature, year);

  return ee.Feature(
    feature.geometry(),
    withMeta
      .combine(feature.toDictionary(), true)
      .combine(seasonal, true)
      .combine(seasonalSmap, true)
      .combine(weatherSummary, true)
      .combine(staticSummary, true)
      .combine(aafcCrop, true)
  );
}

function buildFeaturesForYear(year) {
  return quarterSections.map(function(feature) {
    return buildYearFeature(feature, year);
  });
}

quarterSections = quarterSections.map(addCoverageDiagnostics)
  .filter(ee.Filter.gte('coverage_eligible_year_count', COVERAGE_REQUIRED_YEARS));

var years = ee.List.sequence(START_YEAR, END_YEAR);
var featureTable = ee.FeatureCollection(
  years.iterate(function(year, acc) {
    year = ee.Number(year);
    acc = ee.FeatureCollection(acc);
    return acc.merge(buildFeaturesForYear(year));
  }, ee.FeatureCollection([]))
);

print('Quarter sections after coverage filter', quarterSections.size());
print('Coverage required years', COVERAGE_REQUIRED_YEARS);
print('Feature table preview', featureTable.limit(5));

Export.table.toDrive({
  collection: featureTable,
  description: EXPORT_DESCRIPTION,
  folder: EXPORT_FOLDER,
  fileNamePrefix: EXPORT_FILE_PREFIX,
  fileFormat: 'CSV'
});
