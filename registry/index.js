var oc = require('oc');

var configuration = {
  verbosity: 0,
  baseUrl: 'http://localhost/',
  port: 6000,
  tempDir: './temp/',
  refreshInterval: 600,
  pollingInterval: 5,
  s3: {
    key: '',
    secret: '',
    bucket: 'occomp1',
    region: 'ap-southeast-1',
    path: '//s3.amazonaws.com/occomp1/',
    componentsDir: 'components'
  },
  env: { name: 'production' }
};

var registry = new oc.Registry(configuration);

registry.start(function(err, app){
  console.log('registry has started...')  
  if(err){
    console.log('Registry not started: ', err);
    process.exit(1);
  }
});

