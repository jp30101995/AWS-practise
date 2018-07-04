var registry = new oc.Registry(configuration);

registry.register({
  name: 'getFeatureSwitch',
  register: require('./oc-plugins/hobknob'),
  options: {
      connectionString: connectionString
  }
});