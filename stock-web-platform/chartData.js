import {Line, mixins} from 'vue-chartjs'
const { reactiveProp } = mixins

export var LineChart = {
    extends: Line,
    mixins: [reactiveProp],
    props: {
    	'options': {
    		required: true
    	}
    },
    mounted: function () {
    	this.renderChart(this.chartData, this.options)
    }
}
