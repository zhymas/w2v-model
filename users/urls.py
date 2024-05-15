from django.urls import path
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'records', views.RecordViewSet)
router.register(r'users', views.UserViewSet)

urlpatterns = router.urls
