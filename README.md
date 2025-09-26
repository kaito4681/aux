# aux

## 実験1: Vitにauxを追加

* **--aux all**
`loss = main_loss + 0.1 * aux_loss.sum`
* **--aux mid**
`loss = main_loss + 0.3 * mid_aux_loss`
* **--aux none**
`loss = main_loss`
