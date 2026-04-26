using LSTM_DATA_SHAPER;
using LSTM_DATA_SHAPER.Models;
using System.Text.Json;

namespace LSTM_DATA_SHAPER
{
    public class Program
    {
        private static readonly List<string> MAG7_TICKERS = new()
        {
            "AAPL.US", "MSFT.US", "GOOGL.US", "AMZN.US", "NVDA.US", "META.US", "TSLA.US"
        };

        public static async Task Main(string[] args)
        {
            string rootPath = @"C:\Users\RudyChemik\Desktop\stock_data";

            if (!Directory.Exists(rootPath))
            {
                Console.WriteLine($"RROT NON EXISTING --- {rootPath}");
                return;
            }

            foreach (var ticker in MAG7_TICKERS)
            {
                string folderPath = Path.Combine(rootPath, ticker);

                if (!Directory.Exists(folderPath))
                {
                    Console.WriteLine($"SKIPPED {ticker} --- NOT FOUND");
                    continue;
                }              

                try
                {
                    Console.WriteLine($"\n--- {ticker} --- {ticker} ---");
                    await ProcessTicker(folderPath, ticker);
                    Console.WriteLine($"\n--- DONE ---");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\n --- {ticker} FAILED --- {ex.Message}");
                }
            }
        }

        private static async Task ProcessTicker(string folderPath, string ticker)
        {
            string file5m = Path.Combine(folderPath, ticker + "_5m.txt");
            if (!File.Exists(file5m))
            { 
                throw new FileNotFoundException(); 
            }

            var stockData = new StockData
            {
                FolderPath = folderPath,
                Ticker = ticker,
                Data = JsonSerializer.Deserialize<List<Candle>>(File.ReadAllText(file5m)) ?? new List<Candle>()
            };

            if (stockData.Data.Count == 0)
            {
                throw new InvalidDataException("No candles in 5m data");
            }

            var filler = new CandleFiller(new List<StockData> { stockData });
            filler.Fill();

            var shaper = new CandleShaper();
            await shaper.ShapeData(new List<StockData> { stockData });


            string file15m = Path.Combine(folderPath, ticker + "_15m.txt");

            var record = new TimedStockData
            {
                FolderPath = folderPath,
                Ticker = ticker,
                Data_5M = JsonSerializer.Deserialize<List<Candle>>(File.ReadAllText(file5m)) ?? new List<Candle>(),
                Data_15M = JsonSerializer.Deserialize<List<Candle>>(File.ReadAllText(file15m)) ?? new List<Candle>()
            };

            record.Data_5M = record.Data_5M.OrderBy(c => c.Timestamp).ToList();
            record.Data_15M = record.Data_15M.OrderBy(c => c.Timestamp).ToList();

            var technicals = new TechnicalCalculation(record);
            technicals.CalculateTechnicals();

            var labels = new LabelCalculator(record);
            labels.CalculateLabelsTf();


            //techs skip
            var data5m = record.Data_5M.OrderBy(a => a.Timestamp).Skip(30).ToList();
            var data15m = record.Data_15M.OrderBy(a => a.Timestamp).Skip(30).ToList();

            var out5m = Path.Combine(folderPath, "5m");
            var out15m = Path.Combine(folderPath, "15m");
            Directory.CreateDirectory(out5m);
            Directory.CreateDirectory(out15m);

            var cuts = ComputeCutsFrom5M(data5m, 0.70, 0.85);

            SplitAndSaveByCloseTime(data5m, out5m, 300, cuts);
            SplitAndSaveByCloseTime(data15m, out15m, 900, cuts);
        }

        private readonly record struct TimeCuts(long TrainCutCloseTs, long ValCutCloseTs);

        private static TimeCuts ComputeCutsFrom5M(List<Candle> data5m, double trainFrac, double valFrac)
        {
            int tfSeconds5m = 300;

            var candles = data5m.OrderBy(c => c.Timestamp).ToList();
            int count = candles.Count;

            if (count < 10)
                throw new InvalidOperationException("not enoiugh data for cuts");

            int trainEndIdx = Math.Clamp((int)(count * trainFrac), 1, count - 2);
            int valEndIdx = Math.Clamp((int)(count * valFrac), trainEndIdx + 1, count - 1);

            //TAKWE CLOSE TIME INSTEAD OF CANDLE START!
            long trainCut = candles[trainEndIdx].Timestamp + tfSeconds5m;
            long valCut = candles[valEndIdx].Timestamp + tfSeconds5m;

            return new TimeCuts(trainCut, valCut);
        }

        private static void SplitAndSaveByCloseTime(List<Candle> data, string folder, int tfSeconds, TimeCuts cuts)
        {
            if (data == null || data.Count == 0) { return; }

            var ordered = data.OrderBy(c => c.Timestamp).ToList();

            var train = new List<Candle>();
            var val = new List<Candle>();
            var test = new List<Candle>();

            foreach (var candle in ordered)
            {
                long closeTs = candle.Timestamp + tfSeconds;

                if (closeTs <= cuts.TrainCutCloseTs)
                {
                    train.Add(candle);
                }
                else if (closeTs <= cuts.ValCutCloseTs)
                {
                    val.Add(candle);
                }
                else
                {
                    test.Add(candle);
                }
            }

            var options = new JsonSerializerOptions { WriteIndented = true };
            File.WriteAllText(Path.Combine(folder, "training.txt"), JsonSerializer.Serialize(train, options));
            File.WriteAllText(Path.Combine(folder, "validation.txt"), JsonSerializer.Serialize(val, options));
            File.WriteAllText(Path.Combine(folder, "test.txt"), JsonSerializer.Serialize(test, options));
        }
    }
}