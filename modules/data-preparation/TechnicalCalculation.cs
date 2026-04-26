using LSTM_DATA_SHAPER.Models;
using System;
using System.Collections.Generic;

namespace LSTM_DATA_SHAPER
{
    public class TechnicalCalculation
    {
        private const double EPSILON = 1e-12;
        private readonly TimedStockData _record;

        public TechnicalCalculation(TimedStockData record)
        {
            _record = record;
        }

        public void CalculateTechnicals()
        {
            CalculateForSeries(_record.Data_15M);
            CalculateForSeries(_record.Data_5M);
        }

        private void CalculateForSeries(List<Candle> candles)
        {
            List<Candle> realCandles = new List<Candle>();
            for (int i = 0; i < candles.Count; i++)
            {
                if (candles[i] != null && candles[i].MissingFlag != true)
                {
                    realCandles.Add(candles[i]);
                }
            }

            if (realCandles.Count == 0)
            {
                return;
            }

            int candleCount = realCandles.Count;

            double[] closePrices = new double[candleCount];
            double[] highPrices = new double[candleCount];
            double[] lowPrices = new double[candleCount];
            double[] volumes = new double[candleCount];
            long[] timestamps = new long[candleCount];

            for (int i = 0; i < candleCount; i++)
            {
                closePrices[i] = realCandles[i].Close.Value;
                highPrices[i] = realCandles[i].High.Value;
                lowPrices[i] = realCandles[i].Low.Value;
                volumes[i] = realCandles[i].Volume.HasValue ? realCandles[i].Volume.Value : 0.0;
                timestamps[i] = realCandles[i].Timestamp;
            }

            double?[] sma10 = ComputeSma(closePrices, 10);
            double?[] sma20 = ComputeSma(closePrices, 20);
            double?[] sma50 = ComputeSma(closePrices, 50);
            double?[] atr14 = ComputeAtr(highPrices, lowPrices, closePrices, 14);
            double?[] rsi14 = ComputeRsi(closePrices, 14);
            double?[] logReturns = ComputeLogReturn1(closePrices);
            double?[] rollingSkewness20 = ComputeRollingSkewness(logReturns, 20);
            double?[] kaufmanER10 = ComputeKaufmanEr(closePrices, 10);
            double?[] rollingVwap20 = ComputeRollingVwap(highPrices, lowPrices, closePrices, volumes, 20);
            double?[] anchoredVwap = ComputeAnchoredVwapByBucket(highPrices, lowPrices, closePrices, volumes, timestamps, 3600);
            double?[] amihudIlliquidity20 = ComputeAmihudIlliq(closePrices, volumes, 20);

            double?[] adxValues;
            double?[] plusDI;
            double?[] minusDI;
            ComputeAdx(highPrices, lowPrices, closePrices, 14, out adxValues, out plusDI, out minusDI);

            double?[] pvoLine;
            double?[] pvoSignalLine;
            double?[] pvoHistogram;
            ComputePvo(volumes, 12, 26, 9, out pvoLine, out pvoSignalLine, out pvoHistogram);

            double?[] massIndex;
            ComputeMassIndex(highPrices, lowPrices, 9, 25, out massIndex);

            double?[] emaOfAdx50 = ComputeEmaNullable(adxValues, 50);
            double?[] emaOfMass50 = ComputeEmaNullable(massIndex, 50);

            double?[] volumeRateOfChange = new double?[candleCount];
            for (int i = 1; i < candleCount; i++)
            {
                if (volumes[i - 1] > 0)
                {
                    double rawChange = volumes[i] / volumes[i - 1] - 1.0;
                    volumeRateOfChange[i] = Math.Sign(rawChange) * Math.Log(1.0 + Math.Abs(rawChange));
                }
            }

            for (int i = 0; i < candleCount; i++)
            {
                Candle candle = realCandles[i];
                double close = closePrices[i];

                candle.BODYTOCLOSE = (close - candle.Open.Value) / close;
                candle.HIGHTOCLOSE = (candle.High.Value - close) / close;
                candle.LOWTOCLOSE = (close - candle.Low.Value) / close;

                candle.CloseVsSma10 = Ratio(close, sma10[i]);
                candle.CloseVsSma20 = Ratio(close, sma20[i]);
                candle.CloseVsSma50 = Ratio(close, sma50[i]);
                candle.Sma10VsSma20 = Ratio(sma10[i], sma20[i]);
                candle.Sma20VsSma50 = Ratio(sma20[i], sma50[i]);

                if (atr14[i].HasValue)
                {
                    candle.ATRTOCLOSE = atr14[i].Value / close;
                }

                candle.SKEW = rollingSkewness20[i];
                candle.RSI = rsi14[i];
                candle.ADXRATIO = Ratio(adxValues[i], emaOfAdx50[i]);
                candle.KER = kaufmanER10[i];

                if (massIndex[i].HasValue && massIndex[i].Value > 0)
                {
                    double? ratio = Ratio(massIndex[i], emaOfMass50[i]);
                    if (ratio.HasValue)
                    {
                        candle.MASSI = ratio.Value;
                    }
                    else
                    {
                        candle.MASSI = massIndex[i].Value / 25.0;
                    }
                }

                candle.VROCLOG = volumeRateOfChange[i];
                candle.PVOH = pvoHistogram[i];
                candle.CLOSETOVWAP = Ratio(close, rollingVwap20[i]);
                candle.CLOSETOAVWAP = Ratio(close, anchoredVwap[i]);

                if (amihudIlliquidity20[i].HasValue)
                {
                    candle.AILLIQLOG = Math.Log(amihudIlliquidity20[i].Value + EPSILON);
                }

                if (i > 0 && closePrices[i - 1] > 0)
                {
                    candle.LOGRETURN = Math.Log(close / closePrices[i - 1]);
                }
            }
        }


        //techs
        private static double?[] ComputeSma(double[] values, int period)
        {
            int length = values.Length;
            double?[] sma = new double?[length];
            double runningSum = 0;

            for (int i = 0; i < length; i++)
            {
                runningSum += values[i];

                if (i >= period)
                {
                    runningSum -= values[i - period];
                }

                if (i >= period - 1)
                {
                    sma[i] = runningSum / period;
                }
            }

            return sma;
        }

        private static double[] ComputeEma(double[] values, int period)
        {
            double[] ema = new double[values.Length];

            if (values.Length == 0)
            {
                return ema;
            }

            double smoothingFactor = 2.0 / (period + 1.0);
            ema[0] = values[0];

            for (int i = 1; i < values.Length; i++)
            {
                ema[i] = (values[i] - ema[i - 1]) * smoothingFactor + ema[i - 1];
            }

            return ema;
        }

        private static double?[] ComputeEmaNullable(double?[] values, int period)
        {
            int length = values.Length;
            double?[] ema = new double?[length];
            double smoothingFactor = 2.0 / (period + 1.0);

            int firstValidIndex = -1;
            for (int i = 0; i < length; i++)
            {
                if (values[i].HasValue && !double.IsNaN(values[i].Value) && !double.IsInfinity(values[i].Value))
                {
                    firstValidIndex = i;
                    break;
                }
            }

            if (firstValidIndex < 0)
            {
                return ema;
            }

            ema[firstValidIndex] = values[firstValidIndex].Value;

            for (int i = firstValidIndex + 1; i < length; i++)
            {
                if (values[i].HasValue)
                {
                    ema[i] = (values[i].Value - ema[i - 1].Value) * smoothingFactor + ema[i - 1].Value;
                }
                else
                {
                    ema[i] = ema[i - 1];
                }
            }

            return ema;
        }

        private static double ComputeRsiValue(double averageGain, double averageLoss)
        {
            if (averageLoss > 0)
            {
                return 100.0 - 100.0 / (1.0 + averageGain / averageLoss);
            }

            if (averageGain > 0)
            {
                return 100.0;
            }

            return 50.0;
        }

        private static double?[] ComputeRsi(double[] closePrices, int period)
        {
            int length = closePrices.Length;
            double?[] rsi = new double?[length];

            if (length < period + 1)
            {
                return rsi;
            }

            double[] gains = new double[length];
            double[] losses = new double[length];

            for (int i = 1; i < length; i++)
            {
                double priceChange = closePrices[i] - closePrices[i - 1];
                gains[i] = Math.Max(0, priceChange);
                losses[i] = Math.Max(0, -priceChange);
            }

            double averageGain = 0;
            double averageLoss = 0;
            for (int i = 1; i <= period; i++)
            {
                averageGain += gains[i];
                averageLoss += losses[i];
            }
            averageGain /= period;
            averageLoss /= period;

            rsi[period] = ComputeRsiValue(averageGain, averageLoss);

            double wilderFactor = 1.0 / period;
            for (int i = period + 1; i < length; i++)
            {
                averageGain = gains[i] * wilderFactor + averageGain * (1.0 - wilderFactor);
                averageLoss = losses[i] * wilderFactor + averageLoss * (1.0 - wilderFactor);
                rsi[i] = ComputeRsiValue(averageGain, averageLoss);
            }

            return rsi;
        }

        private static double?[] ComputeAtr(double[] highPrices, double[] lowPrices, double[] closePrices, int period)
        {
            int length = highPrices.Length;
            double?[] atr = new double?[length];
            double[] trueRange = new double[length];

            trueRange[0] = highPrices[0] - lowPrices[0];

            for (int i = 1; i < length; i++)
            {
                double highMinusLow = highPrices[i] - lowPrices[i];
                double highMinusPrevClose = Math.Abs(highPrices[i] - closePrices[i - 1]);
                double lowMinusPrevClose = Math.Abs(lowPrices[i] - closePrices[i - 1]);
                trueRange[i] = Math.Max(highMinusLow, Math.Max(highMinusPrevClose, lowMinusPrevClose));
            }

            if (length < period)
            {
                return atr;
            }

            double initialSum = 0;
            for (int i = 0; i < period; i++)
            {
                initialSum += trueRange[i];
            }
            atr[period - 1] = initialSum / period;

            for (int i = period; i < length; i++)
            {
                atr[i] = (atr[i - 1].Value * (period - 1) + trueRange[i]) / period;
            }

            return atr;
        }

        private static void ComputeAdx(double[] highPrices, double[] lowPrices, double[] closePrices, int period,
            out double?[] adxValues, out double?[] plusDI, out double?[] minusDI)
        {
            int length = highPrices.Length;
            adxValues = new double?[length];
            plusDI = new double?[length];
            minusDI = new double?[length];

            if (length < period + 1)
            {
                return;
            }

            double[] plusDM = new double[length];
            double[] minusDM = new double[length];
            double[] trueRange = new double[length];

            trueRange[0] = highPrices[0] - lowPrices[0];

            for (int i = 1; i < length; i++)
            {
                double upMove = highPrices[i] - highPrices[i - 1];
                double downMove = lowPrices[i - 1] - lowPrices[i];

                if (upMove > 0 && upMove > downMove)
                {
                    plusDM[i] = upMove;
                }

                if (downMove > 0 && downMove > upMove)
                {
                    minusDM[i] = downMove;
                }

                trueRange[i] = Math.Max(
                    highPrices[i] - lowPrices[i],
                    Math.Max(
                        Math.Abs(highPrices[i] - closePrices[i - 1]),
                        Math.Abs(lowPrices[i] - closePrices[i - 1])));
            }

            double smoothedTR = 0;
            double smoothedPlusDM = 0;
            double smoothedMinusDM = 0;

            for (int i = 0; i < period; i++)
            {
                smoothedTR += trueRange[i];
                smoothedPlusDM += plusDM[i];
                smoothedMinusDM += minusDM[i];
            }

            double?[] directionalIndex = new double?[length];
            int startIndex = period - 1;

            for (int i = startIndex; i < length; i++)
            {
                if (i > startIndex)
                {
                    smoothedTR = smoothedTR - smoothedTR / period + trueRange[i];
                    smoothedPlusDM = smoothedPlusDM - smoothedPlusDM / period + plusDM[i];
                    smoothedMinusDM = smoothedMinusDM - smoothedMinusDM / period + minusDM[i];
                }

                if (smoothedTR <= 0)
                {
                    continue;
                }

                double plusDIValue = 100.0 * smoothedPlusDM / smoothedTR;
                double minusDIValue = 100.0 * smoothedMinusDM / smoothedTR;
                plusDI[i] = plusDIValue;
                minusDI[i] = minusDIValue;

                double diSum = plusDIValue + minusDIValue;
                if (diSum > 0)
                {
                    directionalIndex[i] = 100.0 * Math.Abs(plusDIValue - minusDIValue) / diSum;
                }
                else
                {
                    directionalIndex[i] = 0;
                }
            }

            double dxSum = 0;
            int validDxCount = 0;
            for (int i = startIndex; i < length && i < startIndex + period; i++)
            {
                if (directionalIndex[i].HasValue)
                {
                    dxSum += directionalIndex[i].Value;
                    validDxCount++;
                }
            }

            if (validDxCount == 0)
            {
                return;
            }

            int adxStartIndex = startIndex + period - 1;
            double previousAdx = dxSum / validDxCount;

            if (adxStartIndex < length)
            {
                adxValues[adxStartIndex] = previousAdx;
            }

            for (int i = adxStartIndex + 1; i < length; i++)
            {
                if (!directionalIndex[i].HasValue)
                {
                    continue;
                }

                previousAdx = (previousAdx * (period - 1) + directionalIndex[i].Value) / period;
                adxValues[i] = previousAdx;
            }
        }

        private static void ComputePvo(double[] volumeData, int fastPeriod, int slowPeriod, int signalPeriod,
            out double?[] pvoLine, out double?[] signalLine, out double?[] histogram)
        {
            int length = volumeData.Length;
            pvoLine = new double?[length];
            signalLine = new double?[length];
            histogram = new double?[length];

            if (length == 0)
            {
                return;
            }

            double[] emaFast = ComputeEma(volumeData, fastPeriod);
            double[] emaSlow = ComputeEma(volumeData, slowPeriod);

            for (int i = 0; i < length; i++)
            {
                if (emaSlow[i] > 0)
                {
                    pvoLine[i] = 100.0 * (emaFast[i] - emaSlow[i]) / emaSlow[i];
                }
            }

            signalLine = ComputeEmaNullable(pvoLine, signalPeriod);

            for (int i = 0; i < length; i++)
            {
                if (pvoLine[i].HasValue && signalLine[i].HasValue)
                {
                    histogram[i] = pvoLine[i].Value - signalLine[i].Value;
                }
            }
        }

        private static void ComputeMassIndex(double[] highPrices, double[] lowPrices,
            int emaPeriod, int sumPeriod, out double?[] massIndex)
        {
            int length = highPrices.Length;
            massIndex = new double?[length];

            if (length == 0)
            {
                return;
            }

            double[] candleRange = new double[length];
            for (int i = 0; i < length; i++)
            {
                candleRange[i] = Math.Max(0, highPrices[i] - lowPrices[i]);
            }

            double[] singleEma = ComputeEma(candleRange, emaPeriod);
            double[] doubleEma = ComputeEma(singleEma, emaPeriod);

            double rollingSum = 0;

            for (int i = 0; i < length; i++)
            {
                double emaRatio = 0;
                if (doubleEma[i] > 0)
                {
                    emaRatio = singleEma[i] / doubleEma[i];
                }

                rollingSum += emaRatio;

                if (i >= sumPeriod)
                {
                    double oldRatio = 0;
                    if (doubleEma[i - sumPeriod] > 0)
                    {
                        oldRatio = singleEma[i - sumPeriod] / doubleEma[i - sumPeriod];
                    }

                    rollingSum -= oldRatio;
                }

                if (i >= sumPeriod - 1)
                {
                    massIndex[i] = rollingSum;
                }
            }
        }

        private static double?[] ComputeLogReturn1(double[] closePrices)
        {
            double?[] logReturns = new double?[closePrices.Length];

            for (int i = 1; i < closePrices.Length; i++)
            {
                if (closePrices[i - 1] > 0 && closePrices[i] > 0)
                {
                    logReturns[i] = Math.Log(closePrices[i] / closePrices[i - 1]);
                }
            }

            return logReturns;
        }
        private static double?[] ComputeRollingVwap(double[] highPrices, double[] lowPrices,
            double[] closePrices, double[] volumes, int period)
        {
            int length = closePrices.Length;
            double?[] vwap = new double?[length];

            double[] tpTimesVolume = new double[length];
            for (int i = 0; i < length; i++)
            {
                double typicalPrice = (highPrices[i] + lowPrices[i] + closePrices[i]) / 3.0;
                tpTimesVolume[i] = typicalPrice * volumes[i];
            }

            double sumTPV = 0;
            double sumVolume = 0;

            for (int i = 0; i < length; i++)
            {
                sumTPV += tpTimesVolume[i];
                sumVolume += volumes[i];

                if (i >= period)
                {
                    sumTPV -= tpTimesVolume[i - period];
                    sumVolume -= volumes[i - period];
                }

                if (i >= period - 1 && sumVolume > 0)
                {
                    vwap[i] = sumTPV / sumVolume;
                }
            }

            return vwap;
        }

        private static double?[] ComputeAnchoredVwapByBucket(double[] highPrices, double[] lowPrices,
            double[] closePrices, double[] volumes, long[] timestamps, long bucketSizeSeconds)
        {
            int length = closePrices.Length;
            double?[] anchoredVwap = new double?[length];

            long previousBucket = long.MinValue;
            double cumulativeTPV = 0;
            double cumulativeVolume = 0;

            for (int i = 0; i < length; i++)
            {
                long currentBucket = 0;
                if (bucketSizeSeconds > 0)
                {
                    currentBucket = timestamps[i] / bucketSizeSeconds;
                }

                if (currentBucket != previousBucket)
                {
                    previousBucket = currentBucket;
                    cumulativeTPV = 0;
                    cumulativeVolume = 0;
                }

                double typicalPrice = (highPrices[i] + lowPrices[i] + closePrices[i]) / 3.0;
                cumulativeTPV += typicalPrice * volumes[i];
                cumulativeVolume += volumes[i];

                if (cumulativeVolume > 0)
                {
                    anchoredVwap[i] = cumulativeTPV / cumulativeVolume;
                }
            }

            return anchoredVwap;
        }

        private static double?[] ComputeAmihudIlliq(double[] closePrices, double[] volumes, int period)
        {
            int length = closePrices.Length;

            double?[] rawAmihud = new double?[length];
            for (int i = 1; i < length; i++)
            {
                double dollarVolume = closePrices[i] * volumes[i];

                if (closePrices[i - 1] > 0 && closePrices[i] > 0 && dollarVolume > 0)
                {
                    double absReturn = Math.Abs(Math.Log(closePrices[i] / closePrices[i - 1]));
                    rawAmihud[i] = absReturn / (dollarVolume + EPSILON);
                }
            }

            double?[] smoothedIlliquidity = new double?[length];
            double windowSum = 0;
            int validCount = 0;

            for (int i = 0; i < length; i++)
            {
                if (rawAmihud[i].HasValue)
                {
                    windowSum += rawAmihud[i].Value;
                    validCount++;
                }

                if (i >= period && rawAmihud[i - period].HasValue)
                {
                    windowSum -= rawAmihud[i - period].Value;
                    validCount--;
                }

                if (i >= period - 1 && validCount == period)
                {
                    smoothedIlliquidity[i] = windowSum / period;
                }
            }

            return smoothedIlliquidity;
        }
        private static double?[] ComputeRollingSkewness(double?[] values, int windowSize)
        {
            int length = values.Length;
            double?[] skewness = new double?[length];

            if (windowSize < 3)
            {
                return skewness;
            }

            for (int i = windowSize - 1; i < length; i++)
            {
                double windowSum = 0;
                bool windowComplete = true;

                for (int j = i - windowSize + 1; j <= i; j++)
                {
                    if (!values[j].HasValue)
                    {
                        windowComplete = false;
                        break;
                    }

                    windowSum += values[j].Value;
                }

                if (!windowComplete)
                {
                    continue;
                }

                double mean = windowSum / windowSize;
                double secondMoment = 0;
                double thirdMoment = 0;

                for (int j = i - windowSize + 1; j <= i; j++)
                {
                    double deviation = values[j].Value - mean;
                    secondMoment += deviation * deviation;
                    thirdMoment += deviation * deviation * deviation;
                }

                secondMoment /= windowSize;
                thirdMoment /= windowSize;

                if (secondMoment <= EPSILON)
                {
                    continue;
                }

                double biasCorrectionFactor = Math.Sqrt((double)windowSize * (windowSize - 1)) / (windowSize - 2);
                skewness[i] = biasCorrectionFactor * (thirdMoment / Math.Pow(secondMoment, 1.5));
            }

            return skewness;
        }

        private static double?[] ComputeKaufmanEr(double[] closePrices, int period)
        {
            double?[] efficiencyRatio = new double?[closePrices.Length];

            for (int i = period; i < closePrices.Length; i++)
            {
                double netChange = Math.Abs(closePrices[i] - closePrices[i - period]);

                double totalVolatility = 0;
                for (int j = i - period + 1; j <= i; j++)
                {
                    totalVolatility += Math.Abs(closePrices[j] - closePrices[j - 1]);
                }

                if (totalVolatility > EPSILON)
                {
                    efficiencyRatio[i] = Math.Max(0, Math.Min(1, netChange / totalVolatility));
                }
            }

            return efficiencyRatio;
        }

        private static double? Ratio(double? numerator, double? denominator)
        {
            if (numerator.HasValue && denominator.HasValue && denominator.Value > 0)
            {
                return numerator.Value / denominator.Value;
            }

            return null;
        }
    }
}