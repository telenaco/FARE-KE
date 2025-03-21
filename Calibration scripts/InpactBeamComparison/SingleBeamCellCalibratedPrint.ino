#include "MCP356x.h"

/** @brief Pins Configuration */
const uint8_t SDI_PIN         = 11;
const uint8_t SDO_PIN         = 12;
const uint8_t SCK_PIN         = 13;
const uint8_t MCP_ADC_CS_PIN  = 2;
const uint8_t MCP_ADC_IRQ_PIN = 3;
const uint8_t MCLK_PIN        = 0;

/** @brief Pins Configuration */
const MCP356xChannel CHANNEL_A = MCP356xChannel::DIFF_A;


/** @brief MCP356x ADC object */
MCP356x mcpScale(MCP_ADC_IRQ_PIN, MCP_ADC_CS_PIN, MCLK_PIN);

const int NUM_CHANNELS = 1;
struct MCPData {
    int32_t A = 0;
};

MCPData mcpReadings;

void setupADC(MCP356x &adc) {
    adc.setOption(MCP356X_FLAG_USE_INTERNAL_CLK);
    adc.init(&SPI);
    adc.setScanChannels(NUM_CHANNELS, CHANNEL_A);
    adc.setOversamplingRatio(MCP356xOversamplingRatio::OSR_128);
    adc.setGain(MCP356xGain::GAIN_1);
    adc.setADCMode(MCP356xADCMode::ADC_CONVERSION_MODE);
}

void printOutput() {
    unsigned long elapsedTime = micros();
    float gramsForce = mcpScale.getGramsForce(CHANNEL_A);
    char  output[50];
    snprintf(output, sizeof(output), "%lu, %f", elapsedTime, gramsForce);

    Serial.println(output);
    Serial.flush();
}

void setup() {
    Serial.begin(115200);

    SPI.setSCK(SCK_PIN);
    SPI.setMISO(SDO_PIN);
    SPI.setMOSI(SDI_PIN);
    SPI.begin();

    setupADC(mcpScale);

    // Set polynomial calibration
    // mcpScale.setPolynomialCalibration(CHANNEL_A, -1.143e-13, 0.0003968, 261.8);    // Degraw calibration 
    // mcpScale.setPolynomialCalibration(CHANNEL_A, -2.923e-13, 0.0005596, 513.9);     // TAL220 calibration
    mcpScale.setPolynomialCalibration(CHANNEL_A, -5.538e-14, 0.0003295, -227.8);   // CLZ635 calibration

    // Set the conversion mode to POLYNOMIAL 
    mcpScale.setConversionMode(CHANNEL_A, MCP356xConversionMode::POLYNOMIAL); 
}

void loop() {

    if (mcpScale.isr_fired && mcpScale.read() == 2) {
        mcpReadings.A = mcpScale.value(CHANNEL_A);

        printOutput();
    }
}
