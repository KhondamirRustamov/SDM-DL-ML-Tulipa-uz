library(maptools) # Vector data management (sp)
library(raster) # Raster data management
library(rgdal) # Geospatial Data Abstraction Library
library(rgbif) # One of many GBIF access points (Global Biodiversity Information Facility)
library(maps) # Easy access to basic map layers
library(biomod2) # Ensemble SDM package


# download pseudo-absence data
gbif.PIPO.OR.absent <- read.table('data/pseudo_absence_rsep.csv', header=TRUE, sep=',')
colnames(gbif.PIPO.OR.absent) <- c('name', 'lon', 'lat')
coordinates(gbif.PIPO.OR.absent) <- ~ lon + lat
gbif.PIPO.OR.absent <- SpatialPointsDataFrame(coords = coordinates(gbif.PIPO.OR.absent),
                                              data = data.frame(gbif.PIPO.OR.absent))


# download present data
gbif.PIPO.OR.present <- read.table('data/present.csv', header=TRUE, sep=',')
colnames(gbif.PIPO.OR.present) <- c('name', 'lon', 'lat' )
coordinates(gbif.PIPO.OR.present) <- ~ lon + lat
gbif.PIPO.OR.present <- SpatialPointsDataFrame(coords = coordinates(gbif.PIPO.OR.present),
                                       data = data.frame(gbif.PIPO.OR.present))

# visualizing present and pa points on the Uzbekistan map
map('world', xlim = c(55, 75), ylim=c(36,47))
points(gbif.PIPO.OR.absent, pch=21, bg='dodgerblue')
points(gbif.PIPO.OR.present, pch=21, bg='red')

gbif.PIPO.OR.present@data <- data.frame(present = rep(1, nrow(gbif.PIPO.OR.present)))
gbif.PIPO.OR.absent@data <- data.frame(present = rep(0, nrow(gbif.PIPO.OR.absent)))

PIPO.dat <- rbind(gbif.PIPO.OR.present, gbif.PIPO.OR.absent)


# dowbload worldclim data
files <- list.files(path='D:/DIVA-GIS/wc2.1_30s_bio', pattern='tif', full.names=TRUE)
predictors <- stack(files)

# crop worldclim data
ext <- extent(65, 75, 36, 44)
predictors <- crop(predictors, ext)
predictors[is.nan(predictors)] <- 0

env.dat <- extract(x = predictors, y=PIPO.dat)

PIPO.mod.dat <- BIOMOD_FormatingData(resp.var = PIPO.dat,
                                     expl.var = stack(predictors),
                                     resp.name = 'Tulipa affinis')

# modeling
PIPO.mod <- BIOMOD_Modeling(data = PIPO.mod.dat,
                            models = c('GLM', 'GBM', 'RF', 'MARS', 'CTA', 'SRE', 'FDA', 'MAXENT.Phillips'),
                            SaveObj = TRUE,
                            NbRunEval=5,
                            DataSplit=80,
                            VarImport=0)

# SDM performance evaluating
PIPO.mod.eval <- get_evaluations(PIPO.mod)
print(PIPO.mod.eval)

# create prediction maps using all models and save in .grd file
myBiomodProj <- BIOMOD_Projection(modeling.output = PIPO.mod,
                                  new.env = stack(predictors), # modern environment
                                  proj.name = 'current',
                                  selected.models = 'all',
                                  binary.meth = 'TSS',
                                  compress = 'xz' ,
                                  clamping.mask = F,
                                  do.stack = F,
                                  output.format = '.grd')

# save predictions
write.table(get_predictions(PIPO.mod), file='D:/DIVA-GIS/ML prediction/Tulipa_affinis/test1_af.csv', row.names=FALSE)

