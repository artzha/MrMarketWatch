/*global document, window, alert, console, require, config*/
var requirejs = require('requirejs');
var stockData = require('./src/assets/stock.json');
var Pusher = require('pusher');
const av = require('alphavantage')({key: '89RBTI0KIUM8F1JV', datatype: 'json', outputsize: 'compact'});

var pusher = new Pusher({
        appId: '536045',
        key: '5f549e8be14a34c859b2',
        secret: '1fbb36146a7c3e9ee24c',
        cluster: 'us2',
        encrypted: true
    });

var comp_dict = ['GOOG', 'AAPL', 'MSFT', 'AMZN', 'INTC', 'TSLA'];

// formats strings similarly to python
String.prototype.format = function () {
    'use strict';
    var args = arguments;
    return this.replace(/\{\{|\}\}|\{(\d+)\}/g, function (curlyBrack, index) {
        return ((curlyBrack === "{{") ? "{" : ((curlyBrack === "}}") ? "}" : args[index]));
    });
};

setInterval(function () {
    "use strict";
    av.data.batch(comp_dict).then( 
        data => {
            console.log(data['Stock Quotes'][0]['4. timestamp']);
            pusher.trigger('trade', 'stock', data['Stock Quotes']);
        }).catch((err) => {
            // Handle error here
            console.log('a problem happened')
        })
}, 4000);
