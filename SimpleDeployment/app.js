const express = require('express'),
app = express(),
port = process.env.PORT || 3300,
hbs = require('express-handlebars'),
server = require('http').createServer(app),
Twit = require('twit'),
io = require('socket.io')(server);

app.engine('handlebars', hbs());
app.set('view engine', 'handlebars');

app.get('/', function(req, res){
    res.render('home');
});

server.listen(port);

const twitter = Twit({
    consumer_key: 'mxMeVL5oC3jybhhCcUrOjOxaY',
    consumer_secret: 'CbsjfrTmf75O0pQ2rB53bMeJb9uyNwQ2wqdiWq6mjVbW2aNW5M',
    access_token: '2272827588-rflW83knzswfDPb2515c1Fgx95cjrrRhzpuSToc',
    access_token_secret: 'jXInf917dm3DABaDjE5XJDMrsyghIkhO6tPphb4DKUTPb'
});

const stream = twitter.stream('statuses/filter', {track: 'modi'});

io.on('connect', function(socket){
    stream.on('tweet', function(tweet){
        socket.emit('tweets', tweet);
    })
});