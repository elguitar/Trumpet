<template>
  <div id="app">
    <div id="heading">
      <img src="./assets/trump-3123765_640.png" alt="Picture of Trump">
      <h1>Trump speech generator!</h1>
    </div>
    <div id="trumpet_sounds">
    <p>
      {{ text }}
    </p>
    </div>
    <div id="input_text">
      <input v-model="message" placeholder="Add subject">
    </div>
    <div id="controls">
      <select v-model="rage">
        <option disabled value="">Please select rage level</option>
        <option>0.2</option>
        <option>0.7</option>
        <option>1.2</option>
      </select>
      <button v-on:click="generateText">Generate text!</button>
      <button :disabled="isDisabled" v-on:click="generateMoreText">Generate more text!</button>
    </div>
    <div id="about">
    <p>This site is created by Petri Salminen. Source available: LINKHERE</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'App',
  data: function () {
    return {
      message: "",
      rage: 0.2,
      text: "",
    }
  },
  methods: {
    generateText,
    generateMoreText,
  },
  computed: {
    isDisabled,
  }
}

function generateText () {
  axios
    .post('/api/play_trumpet', {'payload': this.message.toLowerCase(), 'temp': this.rage})
    .then(response => (this.text = response.data.text))
}

function generateMoreText () {
  const ending_length = 10
  const ending = this.text.substring(this.text.length - ending_length).toLowerCase();
  axios
    .post('/api/play_trumpet',{
    'payload': ending,
    'temp': this.rage})
    .then(response => (this.text = this.text.slice(0,-10) + response.data.text))
}

function isDisabled () {
  return this.text.length == 0
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
#app #trumpet_sounds p {
  max-width: 50%;
  text-align: center;
  display: inline-block;
}
#app #heading img {
  width: 100%;
  max-width: 500px;
  height: auto;
}
</style>
