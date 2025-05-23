// core_burn/src/kv_cache.rs

#![warn(missing_docs, clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(unsafe_code, clippy::unwrap_used, clippy::expect_used)]

use burn::{
    module::Module, // Для #[derive(Module)], если KVCache будет частью модели Burn.
    tensor::{backend::Backend, Dim, Shape, Tensor}, // Основные типы тензоров.
};
use std::sync::{Arc, Mutex, MutexGuard}; // Для потокобезопасного разделяемого состояния.

/// Реэкспортируем `MhaCache` из Burn, так как он используется для хранения ключей и значений.
/// `MhaCache` - это простая структура, содержащая `key: Tensor<B, 4>` и `value: Tensor<B, 4>`.
pub use burn::nn::attention::MhaCache;

/// Ошибки, которые могут возникнуть при работе с KV-кэшем.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)] // Используем Clone, PartialEq, Eq для удобства тестирования.
pub enum KVCacheError {
    /// Попытка добавить данные в позицию, выходящую за пределы максимальной длины кэша.
    #[error("Позиция {position} выходит за пределы максимальной длины кэша {max_len}.")]
    PositionOutOfBounds { position: usize, max_len: usize },

    /// Не удалось захвати_lockить мьютекс для доступа к внутреннему состоянию кэша.
    /// Обычно это происходит, если другой поток запаниковал, удерживая мьютекс (PoisonError).
    #[error("Не удалось захватить мьютекс для доступа к KV-кэшу: {0}")]
    LockFailed(String),

    /// Попытка добавить тензор с формой, несовместимой с текущим состоянием кэша
    /// (например, другой batch_size, num_heads или head_dim).
    #[error("Несовместимая форма тензора для KV-кэша: ожидалась форма, совместимая с {expected_dims:?}, получена {actual_dims:?}")]
    IncompatibleShape {
        /// Ожидаемые размерности (может быть частично указана, например, для batch, heads, head_dim).
        expected_dims: Vec<usize>,
        /// Фактические размерности полученного тензора.
        actual_dims: Vec<usize>,
    },

    /// Ошибка, возникшая при операциях с тензорами Burn внутри KV-кэша.
    #[error("Ошибка тензора Burn в KV-кэше: {0}")]
    BurnTensor(String), // Оборачиваем ошибку тензора Burn.
}

/// Структура для хранения KV-кэша одного слоя модели-трансформера.
///
/// Позволяет инкрементально обновлять и извлекать тензоры Key (K) и Value (V).
/// Использует `Arc<Mutex<...>>` для обеспечения потокобезопасного доступа и изменения,
/// что делает его пригодным для использования в многопоточных сценариях инференса.
#[derive(Debug, Clone, Module)] // Clone необходим, чтобы передавать кэш между вызовами forward слоев.
pub struct KVCache<B: Backend> {
    /// Внутреннее хранилище для тензоров K и V, обернутое в `Arc<Mutex>` для потокобезопасности.
    /// `MhaCache` содержит `key: Tensor<B, 4>` и `value: Tensor<B, 4>`.
    /// Ожидаемая форма тензоров: `[batch_size, num_heads, sequence_length, head_dim]`.
    cache_data: Arc<Mutex<MhaCache<B>>>,
    /// Максимальная длина последовательности (max_seq_len), которую может хранить этот кэш.
    max_len: usize,
    /// Текущая заполненная длина последовательности в кэше. Также обернута в `Arc<Mutex>`.
    current_len: Arc<Mutex<usize>>,
}

impl<B: Backend> KVCache<B> {
    /// Создает новый, пустой KV-кэш для одного слоя.
    ///
    /// # Аргументы
    /// * `batch_size`: Размер батча.
    /// * `num_heads`: Количество голов внимания (для K и V).
    /// * `max_len`: Максимальная длина последовательности, которую будет хранить кэш.
    /// * `head_dim`: Размерность одной головы внимания.
    /// * `device`: Устройство Burn, на котором будут созданы начальные тензоры.
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        max_len: usize,
        head_dim: usize,
        device: &B::Device,
    ) -> Self {
        // Инициализируем кэш "пустыми" тензорами (с sequence_length = 0).
        // Это важно для корректной работы первой операции конкатенации (`cat`).
        let initial_key = Tensor::zeros([batch_size, num_heads, 0, head_dim], device);
        let initial_value = Tensor::zeros([batch_size, num_heads, 0, head_dim], device);
        let mha_cache = MhaCache::new(initial_key, initial_value);

        Self {
            cache_data: Arc::new(Mutex::new(mha_cache)),
            max_len,
            current_len: Arc::new(Mutex::new(0)),
        }
    }

    /// Обновляет KV-кэш новыми значениями `new_key_states` и `new_value_states`.
    ///
    /// Новые состояния конкатенируются к существующим в кэше по оси `sequence_length`.
    ///
    /// # Аргументы
    /// * `new_key_states`: Новый тензор ключей формы `[batch_size, num_heads, new_seq_len, head_dim]`.
    /// * `new_value_states`: Новый тензор значений формы `[batch_size, num_heads, new_seq_len, head_dim]`.
    ///
    /// # Возвращает
    /// Кортеж `(Tensor<B, 4>, Tensor<B, 4>)` с полными (обновленными) K и V из кэша.
    /// Возвращает `KVCacheError` в случае ошибки (например, превышение `max_len`, ошибка блокировки, несовместимость форм).
    pub fn update_and_get(
        &self,
        new_key_states: Tensor<B, 4>,
        new_value_states: Tensor<B, 4>,
    ) -> Result<(Tensor<B, 4>, Tensor<B, 4>), KVCacheError> {
        // Захватываем мьютексы для доступа к данным кэша и его текущей длине.
        let mut cache_guard = self
            .cache_data
            .lock()
            .map_err(|e| KVCacheError::LockFailed(format!("Ошибка блокировки данных кэша: {}", e)))?;
        let mut current_len_guard = self
            .current_len
            .lock()
            .map_err(|e| KVCacheError::LockFailed(format!("Ошибка блокировки длины кэша: {}", e)))?;

        let new_seq_len = new_key_states.dims()[2]; // Длина новой последовательности (ось 2).

        // Проверяем, не превысит ли добавление новых данных максимальную длину кэша.
        if *current_len_guard + new_seq_len > self.max_len {
            // TODO: В будущем здесь можно реализовать стратегию вытеснения (например, "скользящее окно").
            // Пока просто возвращаем ошибку.
            return Err(KVCacheError::PositionOutOfBounds {
                position: *current_len_guard + new_seq_len,
                max_len: self.max_len,
            });
        }

        // Проверяем совместимость форм новых тензоров с существующими в кэше
        // (кроме размерности sequence_length, которая будет изменяться).
        let current_key_dims = cache_guard.key.dims();
        let new_key_dims = new_key_states.dims();

        // Сравниваем batch_size, num_heads, head_dim.
        if current_key_dims[0] != new_key_dims[0] // batch_size
            || current_key_dims[1] != new_key_dims[1] // num_heads
            // current_key_dims[2] - это seq_len, он меняется
            || current_key_dims[3] != new_key_dims[3] // head_dim
            // Аналогичные проверки для new_value_states, если они могут отличаться от new_key_states.
            || new_key_dims[0] != new_value_states.dims()[0]
            || new_key_dims[1] != new_value_states.dims()[1]
            || new_key_dims[3] != new_value_states.dims()[3]
        {
            return Err(KVCacheError::IncompatibleShape {
                expected_dims: vec![current_key_dims[0], current_key_dims[1], usize::MAX, current_key_dims[3]], // usize::MAX как плейсхолдер для seq_len
                actual_dims: new_key_dims.into(),
            });
        }

        // Конкатенируем новые ключи и значения к существующим.
        // Ось 2 соответствует размерности sequence_length.
        let updated_k = cache_guard.key.clone().cat(vec![new_key_states], 2)
            .map_err(|e| KVCacheError::BurnTensor(format!("Ошибка конкатенации ключей: {}", e)))?;
        let updated_v = cache_guard.value.clone().cat(vec![new_value_states], 2)
            .map_err(|e| KVCacheError::BurnTensor(format!("Ошибка конкатенации значений: {}", e)))?;

        // Обновляем текущую длину кэша.
        *current_len_guard += new_seq_len;

        // Обновляем тензоры K и V внутри мьютекса.
        cache_guard.key = updated_k.clone();
        cache_guard.value = updated_v.clone();

        // Возвращаем клоны обновленных тензоров.
        Ok((updated_k, updated_v))
    }

    /// Возвращает текущее содержимое KV-кэша (ключи и значения) без его изменения.
    ///
    /// # Возвращает
    /// Кортеж `(Tensor<B, 4>, Tensor<B, 4>)` или `KVCacheError` при ошибке блокировки.
    pub fn get_current_values(&self) -> Result<(Tensor<B, 4>, Tensor<B, 4>), KVCacheError> {
        let cache_guard = self
            .cache_data
            .lock()
            .map_err(|e| KVCacheError::LockFailed(format!("Ошибка блокировки данных кэша при чтении: {}", e)))?;
        Ok((cache_guard.key.clone(), cache_guard.value.clone()))
    }

    /// Возвращает текущую заполненную длину последовательности в кэше.
    pub fn current_seq_length(&self) -> Result<usize, KVCacheError> {
        let len_guard = self
            .current_len
            .lock()
            .map_err(|e| KVCacheError::LockFailed(format!("Ошибка блокировки длины кэша при чтении: {}", e)))?;
        Ok(*len_guard)
    }

    /// Возвращает максимальную длину последовательности, которую может хранить кэш.
    pub fn max_sequence_length(&self) -> usize {
        self.max_len
    }

    /// Очищает KV-кэш, сбрасывая его к начальному состоянию (пустые тензоры, current_len = 0).
    ///
    /// # Аргументы
    /// * `device`: Устройство Burn, на котором будут созданы новые пустые тензоры.
    ///             Необходимо, так как тензоры не хранят информацию об устройстве после создания.
    pub fn reset(&self, device: &B::Device) -> Result<(), KVCacheError> {
        let mut cache_guard = self
            .cache_data
            .lock()
            .map_err(|e| KVCacheError::LockFailed(format!("Ошибка блокировки данных кэша при сбросе: {}", e)))?;
        let mut current_len_guard = self
            .current_len
            .lock()
            .map_err(|e| KVCacheError::LockFailed(format!("Ошибка блокировки длины кэша при сбросе: {}", e)))?;

        // Получаем размерности из существующего (возможно, непустого) ключа,
        // чтобы создать новый пустой тензор с правильными batch_size, num_heads, head_dim.
        let dims = cache_guard.key.dims();
        let batch_size = dims[0];
        let num_heads = dims[1];
        // seq_len будет 0
        let head_dim = dims[3];

        cache_guard.key = Tensor::zeros([batch_size, num_heads, 0, head_dim], device);
        cache_guard.value = Tensor::zeros([batch_size, num_heads, 0, head_dim], device);
        *current_len_guard = 0;

        Ok(())
    }
}
