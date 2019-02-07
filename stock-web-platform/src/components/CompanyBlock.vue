<template>
<div class="company-card">
    <div class="company-chart">
        <b>{{name}}</b>
        <div class="company-forecast">
            <b>Forecast</b>
        </div>
        <div class="chart-container">
            <line-chart :width="3" :height="1" :options="updateOptions" :chartData="updateCollection"/>
        </div>
    </div>
    <div>
        <b>
            Predictions
        </b>
        <div>
            Predicted Selling Price: 130.80
        </div>
    </div>
</div>
</template>

<script>
/*  global document, alert, console, require, Math */
import {LineChart} from '../../chartData.js'

export default {
    name: 'company-block',
    props: {
        'name': {
            type: String,
            required: true
        },
        'chartData': {
            required: true
        }
    },
    components: {
        'line-chart': LineChart
    },
    data: function () {
        return {
            chartOptions: {
                responsive: true,
                title: {
                    display: true,
                    text: 'Stock Prices'
                },
                tooltips: {
                    mode: 'index',
                    intersect: false
                },
                hover: {
                    mode: 'nearest',
                    intersect: true
                },
                scales: {
                    xAxes: [
                        {
                            display: true,
                            scaleLabel: {
                                display: true,
                                labelString: 'Time  '
                            }
                        }
                    ],
                    yAxes: [
                        {
                            display: true,
                            scaleLabel: {
                                display: true,
                                labelString: 'Price'
                            }
                        }
                    ]
                },
                animation: false
            },
            dataUpdated: null,
            yMin: 0,
            yMax: 0
        }
    },
    mounted: function () {
        this.dataUpdated = false
    },
    computed: {
        updateCollection: function () {
            if (this.chartData) {
                return this.fillData()
            }
        },
        updateOptions: function () {
            return this.chartOptions
        }
    },
    methods: {
        fillData: function () {
            return {
                labels: this.chartData.dates,
                datasets: [
                    {
                        label: this.name,
                        backgroundColor: 'rgb(125, 195, 242)',
                        borderColor: 'rgb(54, 162, 235)',
                        data: this.chartData.prices || []
                    }
                ]
            }
        }
    }
}
</script>

<style>
.company-card {
    border-style: solid;
    border-width: 3px;
    border-radius: 20px;
    border-color: #F2F5F8;
    padding: 10px;
    width: 100%;
    height: 20%;
    margin: 20px;
}
.chart-container {
    position: relative;
    width: 100%;
}
.company-chart {
    display: inline;
    margin: auto;
    width: 50%;
}
.company-forecast {
    display: inline;
    margin: auto;
    width: 50%;
}
</style>
