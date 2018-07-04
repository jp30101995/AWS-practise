var connection,
    client = require('./hobknob-client');

module.exports.register = function(options, dependencies, next){
  client.connect(options.connectionString, function(err, conn){
    connection = conn;
    next();
  });
};

module.exports.execute = function(featureName){
  return connection.get(featureName);
};