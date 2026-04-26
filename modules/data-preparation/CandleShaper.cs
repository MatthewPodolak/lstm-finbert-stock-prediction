using LSTM_DATA_SHAPER.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;

namespace LSTM_DATA_SHAPER
{
    public class CandleShaper
    {
        private static readonly TimeZoneInfo ExchangeTimeZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
        private static readonly TimeSpan MarketOpenLocal = new TimeSpan(9, 30, 0);
        private static readonly TimeSpan MarketCloseLocal = new TimeSpan(16, 0, 0);

        public async Task ShapeData(List<StockData> data)
        {
            foreach (var stock in data)
            {
                var sorted = stock.Data.OrderBy(a => a.Timestamp).ToList();
                var session5m = sorted.Where(IsRegularSession).ToList();

                Task save5m = Task.Run(() => JsonFileSave(stock.FolderPath, stock.Ticker, "5m", session5m));
                Task save15m = Task.Run(() => Shape15mins(stock.FolderPath, stock.Ticker, session5m));

                await Task.WhenAll(save5m, save15m);
            }
        }

        private async Task Shape15mins(string folderPath, string ticker, List<Candle> sessionData)
        {
            var timeFrame15 = new List<Candle>();
            var groups = sessionData.GroupBy(c => Get15mBucketStartLocal(c)).OrderBy(g => g.Key);

            foreach (var g in groups)
            {
                var slice = g.OrderBy(c => c.Timestamp).ToList();

                int missing = slice.Count(c => c.MissingFlag ?? false);

                var candle = new Candle
                {
                    GmtOffset = 0,
                    DateTime = g.Key.UtcDateTime.ToString("O"),
                    Timestamp = g.Key.ToUnixTimeSeconds(),
                    Open = slice.First().Open,
                    High = slice.Max(c => c.High),
                    Low = slice.Min(c => c.Low),
                    Close = slice.Last().Close,
                    Volume = slice.Sum(c => c.Volume),
                    MissingFlag = missing / slice.Count >= 2.0 / 3.0,
                };

                timeFrame15.Add(candle);
            }

            await JsonFileSave(folderPath, ticker, "15m", timeFrame15);
        }

        private async Task JsonFileSave(string folderPath, string ticker, string timeFrame, List<Candle> data)
        {
            var fileName = $"{ticker}_{timeFrame}.txt";
            var filePath = Path.Combine(folderPath, fileName);

            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };

            var json = JsonSerializer.Serialize(data, options);
            await File.WriteAllTextAsync(filePath, json);
        }

        private static DateTimeOffset Get15mBucketStartLocal(Candle c)
        {
            var utc = DateTimeOffset.FromUnixTimeSeconds(c.Timestamp);
            var local = TimeZoneInfo.ConvertTime(utc, ExchangeTimeZone);

            var sessionDate = local.Date;
            var timeOfDay = local.TimeOfDay;

            int minutesFromOpen = (int)(timeOfDay - MarketOpenLocal).TotalMinutes;
            int bucketIndex = minutesFromOpen / 15;

            var bucketStartLocal = sessionDate.Add(MarketOpenLocal).Add(TimeSpan.FromMinutes(bucketIndex * 15));

            return new DateTimeOffset(bucketStartLocal, ExchangeTimeZone.GetUtcOffset(bucketStartLocal));
        }
        private static bool IsRegularSession(Candle c)
        {
            var utc = DateTimeOffset.FromUnixTimeSeconds(c.Timestamp);
            var local = TimeZoneInfo.ConvertTime(utc, ExchangeTimeZone);

            return local.TimeOfDay >= MarketOpenLocal && local.TimeOfDay < MarketCloseLocal;
        }
    }
}