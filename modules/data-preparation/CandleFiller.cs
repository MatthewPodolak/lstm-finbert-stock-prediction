using LSTM_DATA_SHAPER.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LSTM_DATA_SHAPER
{
    public class CandleFiller
    {
        private List<StockData> Data;
        public CandleFiller(List<StockData> StockData)
        {
            Data = StockData;
        }

        public void Fill()
        {
            RemoveEmptyFromStart();
            ForwardFill();
        }

        private void RemoveEmptyFromStart()
        {
            foreach (var ticker in Data)
            {
                ticker.Data.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));

                while (ticker.Data.Count > 0 && IsNullCandle(ticker.Data[0]))
                {
                    ticker.Data.RemoveAt(0);
                }
            }
        }

        private void ForwardFill()
        {
            foreach(var ticker in Data)
            {
                Candle? lastValid = null;
                ticker.Data.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));

                for (int i = 0; i < ticker.Data.Count; i++)
                {
                    var c = ticker.Data[i];
                    bool anyMissing = IsNullCandle(c);

                    if (!anyMissing)
                    {
                        c.MissingFlag = false;
                        lastValid = c;
                        continue;
                    }

                    c.MissingFlag = true;
                    if (lastValid == null)
                    {
                        continue;
                    }

                    var prevClose = lastValid.Close;
                    if (prevClose == null)
                    {
                        continue;
                    }

                    c.Open = prevClose;
                    c.High = prevClose;
                    c.Low = prevClose;
                    c.Close = prevClose;
                    c.Volume = 0;
                    c.MissingFlag = true;
                }
            }
        }

        private bool IsNullCandle(Candle c)
        {
            return c.Open == null || c.High == null || c.Low == null || c.Close == null || c.Volume == null;
        }
    }
}