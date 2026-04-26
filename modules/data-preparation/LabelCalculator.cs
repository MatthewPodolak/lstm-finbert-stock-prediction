using LSTM_DATA_SHAPER.Models;
using System;
using System.Collections.Generic;

namespace LSTM_DATA_SHAPER
{
    public class LabelCalculator
    {
        private readonly TimedStockData _record;

        public LabelCalculator(TimedStockData record)
        {
            _record = record;
        }

        public void CalculateLabelsTf()
        {
            double barrierMultiplier5m = 1.5;
            double barrierMultiplier15m = 1.3;

            int baseLookahead5m = 12;
            int baseLookahead15m = 10;

            int minLookahead5m = 4;
            int maxLookahead5m = 24;

            int minLookahead15m = 4;
            int maxLookahead15m = 20;

            double lowVolThreshold5m = 0.0005;
            double highVolThreshold5m = 0.0040;

            double lowVolThreshold15m = 0.0008;
            double highVolThreshold15m = 0.0060;

            CalculateLabels(_record.Data_5M, barrierMultiplier5m, baseLookahead5m, minLookahead5m, maxLookahead5m,
                lowVolThreshold5m, highVolThreshold5m, true
            );

            CalculateLabels(_record.Data_15M, barrierMultiplier15m, baseLookahead15m, minLookahead15m, maxLookahead15m,
                lowVolThreshold15m, highVolThreshold15m, false
            );
        }

#pragma warning disable
        private int GetDynamicLookahead(double atrPercent, int baseLookahead, int minLookahead, int maxLookahead,
            double lowVolThreshold, double highVolThreshold)
        {
            if (atrPercent <= 0 || lowVolThreshold <= 0 || highVolThreshold <= lowVolThreshold)
            {
                return baseLookahead;
            }

            if (atrPercent <= lowVolThreshold)
            {
                return maxLookahead;
            }

            if (atrPercent >= highVolThreshold)
            {
                return minLookahead;
            }

            double interpolation = (atrPercent - lowVolThreshold) / (highVolThreshold - lowVolThreshold);
            double lookahead = maxLookahead + interpolation * (minLookahead - maxLookahead);

            int result = (int)Math.Round(lookahead);

            if (result < minLookahead)
            {
                result = minLookahead;
            }

            if (result > maxLookahead)
            {
                result = maxLookahead;
            }

            return result;
        }

#pragma warning disable
        private void CalculateLabels(List<Candle> candles, double barrierMultiplier, int baseLookahead,
            int minLookahead, int maxLookahead, double lowVolThreshold, double highVolThreshold, bool is5m)
        {
            if (candles == null || candles.Count == 0)
            {
                return;
            }

            int candleCount = candles.Count;

            for (int i = 0; i < candleCount; i++)
            {
                Candle candle = candles[i];

                if (!candle.ATRTOCLOSE.HasValue || candle.ATRTOCLOSE.Value <= 0)
                {
                    SetEventDirection(candle, null, is5m);
                    continue;
                }

                double barrierLogThreshold = barrierMultiplier * candle.ATRTOCLOSE.Value;

                int dynamicLookahead = GetDynamicLookahead(candle.ATRTOCLOSE.Value, baseLookahead, minLookahead,
                    maxLookahead, lowVolThreshold, highVolThreshold
                );

                double upperBarrier = candle.Close.Value * Math.Exp(barrierLogThreshold);
                double lowerBarrier = candle.Close.Value * Math.Exp(-barrierLogThreshold);

                int? label = null;
                bool invalidDataInWindow = false;

                for (int h = 1; h <= dynamicLookahead; h++)
                {
                    int futureIndex = i + h;

                    if (futureIndex >= candleCount)
                    {
                        break;
                    }

                    Candle futureCandle = candles[futureIndex];

                    if (!futureCandle.Close.HasValue || futureCandle.Close.Value <= 0)
                    {
                        invalidDataInWindow = true;
                        break;
                    }

                    double futureClose = futureCandle.Close.Value;

                    if (futureClose >= upperBarrier)
                    {
                        label = 1;
                        break;
                    }

                    if (futureClose <= lowerBarrier)
                    {
                        label = -1;
                        break;
                    }
                }

                if (invalidDataInWindow)
                {
                    SetEventDirection(candle, null, is5m);
                    continue;
                }

                if (label == null)
                {
                    label = 0;
                }

                SetEventDirection(candle, label, is5m);
            }
        }

        private void SetEventDirection(Candle candle, int? value, bool is5m)
        {
            if (is5m)
            {
                candle.EventDirection5m = value;
            }
            else
            {
                candle.EventDirection15m = value;
            }
        }
    }
}