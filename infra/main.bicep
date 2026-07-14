targetScope = 'resourceGroup'

@description('Azure region for the Container Apps environment and both applications.')
param location string = resourceGroup().location

@description('Name of the shared Azure Container Apps environment.')
param environmentName string = 'aircraft-maintenance-env'

@description('Public GHCR image for the FastAPI service, including an immutable tag.')
param apiImage string

@description('Public GHCR image for the Dash dashboard, including an immutable tag.')
param dashboardImage string

@description('Immutable Git revision shown by the API and dashboard.')
param deploymentVersion string

@description('Administrator login for the portfolio metadata database.')
param sqlAdminLogin string = 'simulatoradmin'

@secure()
@description('Administrator password for the portfolio metadata database.')
param sqlAdminPassword string

var storageAccountName = 'st${take(uniqueString(resourceGroup().id), 20)}'
var sqlServerName = 'sql${take(uniqueString(resourceGroup().id), 20)}'

var commonTags = {
  project: 'aircraft-maintenance-ml-simulator'
  environment: 'portfolio'
  dataClassification: 'synthetic'
  costControl: 'scale-to-zero'
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: commonTags
  properties: {
    appLogsConfiguration: {
      destination: 'azure-monitor'
    }
  }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  tags: commonTags
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    isHnsEnabled: true
    minimumTlsVersion: 'TLS1_2'
  }
}

resource dataLakeFileSystem 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  name: '${storage.name}/default/maintenance-lake'
  properties: {
    publicAccess: 'None'
  }
}

var storageConnectionString = 'DefaultEndpointsProtocol=https;AccountName=${storage.name};AccountKey=${storage.listKeys().keys[0].value};EndpointSuffix=${az.environment().suffixes.storage}'

resource sqlServer 'Microsoft.Sql/servers@2023-08-01-preview' = {
  name: sqlServerName
  location: location
  tags: commonTags
  properties: {
    administratorLogin: sqlAdminLogin
    administratorLoginPassword: sqlAdminPassword
    minimalTlsVersion: '1.2'
    publicNetworkAccess: 'Enabled'
    version: '12.0'
  }
}

resource allowAzureServices 'Microsoft.Sql/servers/firewallRules@2023-08-01-preview' = {
  parent: sqlServer
  name: 'AllowAzureServices'
  properties: {
    startIpAddress: '0.0.0.0'
    endIpAddress: '0.0.0.0'
  }
}

resource pipelineDatabase 'Microsoft.Sql/servers/databases@2023-08-01' = {
  parent: sqlServer
  name: 'simulator'
  location: location
  tags: commonTags
  sku: {
    name: 'GP_S_Gen5_1'
    tier: 'GeneralPurpose'
    family: 'Gen5'
    capacity: 1
  }
  properties: {
    autoPauseDelay: 60
    minCapacity: json('0.5')
    maxSizeBytes: 34359738368
  }
}

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'maintenance-simulator-api'
  location: location
  tags: commonTags
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: [
        {
          name: 'storage-connection-string'
          value: storageConnectionString
        }
      ]
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          env: [
            {
              name: 'PIPELINE_STORAGE_CONNECTION_STRING'
              secretRef: 'storage-connection-string'
            }
            {
              name: 'PIPELINE_FILE_SYSTEM'
              value: 'maintenance-lake'
            }
            {
              name: 'APP_VERSION'
              value: deploymentVersion
            }
          ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

resource dashboard 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'maintenance-simulator-dashboard'
  location: location
  tags: commonTags
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      secrets: [
        {
          name: 'storage-connection-string'
          value: storageConnectionString
        }
      ]
      ingress: {
        external: true
        targetPort: 8050
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'dashboard'
          image: dashboardImage
          env: [
            {
              name: 'PIPELINE_STORAGE_CONNECTION_STRING'
              secretRef: 'storage-connection-string'
            }
            {
              name: 'PIPELINE_FILE_SYSTEM'
              value: 'maintenance-lake'
            }
            {
              name: 'APP_VERSION'
              value: deploymentVersion
            }
          ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
        rules: [
          {
            name: 'http'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

resource pipelineJob 'Microsoft.App/jobs@2024-03-01' = {
  name: 'maintenance-simulator-pipeline'
  location: location
  tags: commonTags
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: environment.id
    configuration: {
      triggerType: 'Schedule'
      replicaTimeout: 1800
      replicaRetryLimit: 1
      scheduleTriggerConfig: {
        cronExpression: '0 2 * * *'
        parallelism: 1
        replicaCompletionCount: 1
      }
      secrets: [
        {
          name: 'storage-connection-string'
          value: storageConnectionString
        }
        {
          name: 'sql-admin-password'
          value: sqlAdminPassword
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'pipeline'
          image: apiImage
          command: [
            'python'
            '-m'
            'pipeline.run'
          ]
          env: [
            {
              name: 'PIPELINE_STORAGE_CONNECTION_STRING'
              secretRef: 'storage-connection-string'
            }
            {
              name: 'PIPELINE_FILE_SYSTEM'
              value: 'maintenance-lake'
            }
            {
              name: 'PIPELINE_SQL_SERVER'
              value: '${sqlServer.name}.database.windows.net'
            }
            {
              name: 'PIPELINE_SQL_DATABASE'
              value: pipelineDatabase.name
            }
            {
              name: 'PIPELINE_SQL_USERNAME'
              value: sqlAdminLogin
            }
            {
              name: 'PIPELINE_SQL_PASSWORD'
              secretRef: 'sql-admin-password'
            }
          ]
          resources: {
            cpu: json('0.25')
            memory: '0.5Gi'
          }
        }
      ]
    }
  }
}

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output dashboardUrl string = 'https://${dashboard.properties.configuration.ingress.fqdn}'
output storageAccountName string = storage.name
output sqlServerName string = sqlServer.name
