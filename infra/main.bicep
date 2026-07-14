targetScope = 'resourceGroup'

@description('Azure region for the Container Apps environment and both applications.')
param location string = resourceGroup().location

@description('Name of the shared Azure Container Apps environment.')
param environmentName string = 'aircraft-maintenance-env'

@description('Public GHCR image for the FastAPI service, including an immutable tag.')
param apiImage string

@description('Public GHCR image for the Dash dashboard, including an immutable tag.')
param dashboardImage string

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
      destination: 'none'
    }
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

output apiUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output dashboardUrl string = 'https://${dashboard.properties.configuration.ingress.fqdn}'
