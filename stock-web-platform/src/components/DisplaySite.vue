<template>
<div class="nav">
    <h1>Mr.Stock DIJA</h1>
    <div id="intro-container">
        <p id="intro">
            Mr.Stock DIJA is an etf, short for electronic trading advisor, that helps you profit from the stock market. He weights and evaluates several technical indicators of the stock market to give you advice on what to invest in. Choose companies below and see how well his predictions have been!
        </p>
    </div>
    <div id="search-wrapper">
        <input id="search-bar" type="text"
            v-model="searchQuery" placeholder="Search for company here..." @keyup="submitSearch">
        <button id="search-button" @click="submitSearch">
        </button>
    </div>
    <div>
        <transition-group name="staggered-fade" tag="div">
            <CompanyBlock v-for="(company, index) in filteredCompanies"
                :key="index"
                :name="company.name"
                :chartData="getChartData(company.name)"
                class="company-block-container">
            </CompanyBlock>
        </transition-group>
    </div>
</div>
</template>

<script>
/*  global document, alert, console, require, Math */
import CompanyBlock from './CompanyBlock.vue'
const Axios = require('axios')

export default {
    name: 'display-site',
    components: {
        CompanyBlock
    },
    data: function () {
        return {
            companies: [
                {name: 'AAPL'},
                {name: 'MSFT'},
                {name: 'AMZN'},
                {name: 'INTC'},
                {name: 'TSLA'},
                {name: 'GOOG'}
            ],
            searchQuery: '',
            isCompany: false,
            stockApiUrlBase: 'https://api.iextrading.com/1.0'
        }
    },
    computed: {
        filteredCompanies: function () {
            return this.companies.filter((company) => {
                return company.name.toUpperCase().match(this.searchQuery.toUpperCase())
            })
        },
        filteredCompanyNames: function () {
            return this.companies.filter((company) => {
                if (company.name.toUpperCase().match(this.searchQuery.toUpperCase())) {
                    return company.name
                }
            })
        },
        getCompanySymbolsString: function () {
            return this.companies.reduce((total, currentValue, index) => {
                return (index === 1) ? total.name + ',' + currentValue.name : total + ',' + currentValue.name
            })
        }
    },
    created: function () {
        Axios.get(this.genBatchStockApiUrl(this.getCompanySymbolsString))
            .then(response => {
                const companies = Object.keys(response.data)
                const fields = Object.keys(response.data[companies[0]])
                companies.forEach(companyName => {
                    for (let index in this.companies) {
                        if (this.companies[index].name === companyName) {
                            for (let field of fields) {
                                this.companies[index][field] = response.data[companyName][field]
                            }
                        }
                    }
                })
                this.companies = Array.from(Object.create(this.companies))
            })
    },
    methods: {
        genBatchStockApiUrl: function (symbols) {
            return this.stockApiUrlBase + '/stock/market/batch?symbols=' + symbols +
                '&types=quote,news,chart&range=1m&last=5'
        },
        removeSearchQuery: function () {
            this.searchQuery = ''
            this.isCompany = false
        },
        submitSearch: function () {
            this.isResult = true
            return this.filteredCompanies
        },
        getChartData: function (symbol) {
            if (this.companies[0].chart) {
                for (let company of this.companies) {
                    if (company.chart && company.name === symbol) {
                        let dates = []
                        let prices = []
                        company.chart.forEach(dataset => {
                            dates.push(dataset.date)
                            prices.push(dataset.open)
                        })
                        return {
                            'dates': dates,
                            'prices': prices
                        }
                    }
                }
            } else {
                return []
            }
        }
    }
}
</script>

<style scoped>
h1, h2 {
    text-align: center;
    font-weight: normal;
}
ul {
    list-style-type: none;
    padding: 0;
}
li {
    display: inline-block;
    margin: 0 10px;
}
a {
    color: #42b983;
}
#intro-container {
    width: 60%;
    padding: 10px;
    margin-left: auto;
    margin-right: auto;
    background-color: #F2F5F8;
    border-radius: 10px;
}
#intro {
    width: 100%;
}

.nav {
    margin: 10px;
}

/* styles for search bar */
#search-bar{
    height: 30px;
    width: 20%;
    outline: none;
    margin: 10px;
    border: 1px solid rgba(0,0,0,.12);
    transition: .15s all ease-in-out;
    background: white;
}
#search-button{
    height: 30px;
    width: 30px;
    padding: 3px;
    background-color: white;
    outline: none;
    border-radius: 5px;
    border: none;
    background-image: url('../assets/search-icon.svg');
    background-repeat: no-repeat;
    background-size: cover;
}
#search-button:hover{
    cursor: pointer;
    background-color: #9DCDC0;
}

#search-wrapper{
    display: flex;
    align-items: center;
    justify-content: center;
    width: auto;
    height: auto;
    padding: 5px;
}

/* styles for each company block */
.container {
    position: relative;
    display: block;
    height: 30%;
    width: 100%;
}
.company-display{
    width: auto;
    height: 400px;
    display: block;
    background-color: beige;
}
#company-title{
    display: block;
    width: auto;
}

.company-block-container {
    width: 100%;
    height: 20%;
}

/* styles for animating list transitions */
.staggered-fade-move {
    transition: all 1s;
}
.staggered-fade-enter-active, .staggered-fade-leave-active {
  transition: all 1s;
}
.staggered-fade-enter, .staggered-fade-leave-to {
  opacity: 0;
  transform: translateY(30px);
}

</style>
